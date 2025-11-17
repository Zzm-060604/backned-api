# ------------------------------
# 整合所有依赖导入（统一管理）
# ------------------------------
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import datasets.btmri  # BiomedCoOp数据集依赖
import trainers.BiomedCoOp.biomedcoop_biomedclip  # BiomedCoOp模型依赖
import json
import time
import os

# ------------------------------
# 全局常量配置（统一整合，支持多文件夹读取）
# ------------------------------
# 服务基础配置
app = FastAPI(title="BiomedCoOp 推理服务（支持多文件夹上传+完整推理）")

# 日志配置（记录上传/推理全流程）
logging.basicConfig(
    filename="biomedcoop_inference.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(user)s - %(message)s"  # 新增用户维度日志
)
logger = logging.getLogger(__name__)

# 模型与数据核心路径（支持多类别文件夹扫描）
TEMP_ROOT = Path("/home/ooze/BiomedCoOp/data/BTMRI/BTMRI")  # 根目录（存放所有类别子文件夹）
OUTPUT_JSON_PATH = Path("/home/ooze/BiomedCoOp/data/BTMRI/split_BTMRI.json")  # 元数据文件
DATA_ROOT = "/home/ooze/BiomedCoOp/data"
DEFAULT_MODEL_DIR = "output/btmri/run1"  # 默认模型目录
DEFAULT_LOAD_EPOCH = 50  # 默认加载轮次
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}  # 支持的图片格式

# 文件上传配置（支持多文件夹，用户可指定目标子文件夹）
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 单文件最大100MB
DEFAULT_UPLOAD_FOLDER = "temp"  # 默认上传子文件夹（无指定时使用）

# 认证配置（保留原有安全逻辑）
SECRET_KEY = "5f8a8f8d8b6e4a2c9d8f7e6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_PASSWORD_LENGTH = 72

# 跨域配置（支持前端调试）
origins = [
    "http://localhost:5000",
    "http://localhost:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 数据模型定义（适配上传+推理流程）
# ------------------------------
class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    full_name: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None

class UserInDB(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    hashed_password: str
    created_at: datetime
    model_config = ConfigDict(mutable=True)

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    code: int = 200
    token: str
    token_type: str = "bearer"
    message: str = "登录成功"
    user_info: UserResponse

class InferenceParams(BaseModel):
    """推理参数模型（支持自定义模型路径和轮次）"""
    model_dir: Optional[str] = DEFAULT_MODEL_DIR
    load_epoch: Optional[int] = DEFAULT_LOAD_EPOCH
    dataset_config: Optional[str] = "/home/ooze/BiomedCoOp/configs/datasets/btmri.yaml"

class UploadResponse(BaseModel):
    """上传响应模型（包含文件夹和文件详情）"""
    code: int = 200
    msg: str = "文件上传成功"
    data: Dict[str, Any]

class InferenceResponse(BaseModel):
    """推理响应模型（包含预测结果和统计信息）"""
    code: int = 200
    msg: str = "推理成功"
    data: Dict[str, Any]

# ------------------------------
# 内存存储（用户数据管理，保留原有逻辑）
# ------------------------------
fake_db = {
    "users": [],
    "next_id": 1
}

# ------------------------------
# 核心工具函数（整合上传/推理/认证逻辑）
# ------------------------------
def add_user_context_log(record, user: str):
    """日志添加用户上下文（便于追踪操作人）"""
    record.user = user
    return logger.makeRecord(
        record.name, record.levelno, record.pathname, record.lineno,
        record.msg, record.args, record.exc_info, record.funcName,
        record.stack_info, user=user
    )

# 1. 认证相关工具
def get_password_hash(password: str) -> str:
    """密码哈希处理（限制长度防溢出）"""
    if len(password) > MAX_PASSWORD_LENGTH:
        password = password[:MAX_PASSWORD_LENGTH]
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """密码验证（长度兼容处理）"""
    if len(plain_password) > MAX_PASSWORD_LENGTH:
        plain_password = plain_password[:MAX_PASSWORD_LENGTH]
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """生成JWT令牌"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_username(username: str) -> Optional[UserInDB]:
    """通过用户名查询用户"""
    return next((u for u in fake_db["users"] if u.username == username), None)

# 2. 文件处理工具（支持多文件夹）
def secure_filename(filename: str) -> str:
    """安全处理文件名（防路径穿越攻击）"""
    # 移除危险字符（路径分隔符、上级目录标识）
    filename = filename.replace("/", "").replace("\\", "").replace("..", "").strip()
    # 处理空文件名（生成时间戳命名）
    if not filename:
        return f"unknown_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}"
    return filename

def get_upload_target_dir(folder: str) -> Path:
    """获取上传目标目录（不存在则自动创建）"""
    target_dir = TEMP_ROOT / folder
    target_dir.mkdir(exist_ok=True, parents=True)  # 递归创建目录（支持多级）
    return target_dir

def count_uploaded_files(root_dir: Path = TEMP_ROOT) -> int:
    """统计根目录下所有有效图片文件数量（支持多子文件夹）"""
    total = 0
    for child in root_dir.iterdir():
        if child.is_dir():  # 遍历所有子文件夹（类别文件夹）
            for file in child.iterdir():
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    total += 1
    return total

# 3. 模型配置工具（BiomedCoOp核心逻辑）
def extend_cfg(cfg):
    """扩展BiomedCoOp模型配置（补充prompt学习参数）"""
    from yacs.config import CfgNode as CN
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # 使用所有类别
    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"  # 文本提示前缀
    cfg.TRAINER.BIOMEDCOOP.CSC = False  # 关闭类别特异性提示
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"  # 类别token位置
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4  # 上下文token数量
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"  # 精度配置
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0  # SCCM损失权重
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 1.0  # KDSP损失权重
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5  # 温度系数
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50  # 提示数量

def setup_cfg(dataset_config: str) -> Any:
    """加载并初始化模型配置"""
    cfg = get_cfg_default()
    extend_cfg(cfg)
    # 合并数据集和训练器配置
    cfg.merge_from_file(dataset_config)
    cfg.merge_from_file("/home/ooze/BiomedCoOp/configs/trainers/biomedcoop.yaml")
    cfg.DATASET.ROOT = DATA_ROOT
    cfg.TRAINER.NAME = "BiomedCoOp_BiomedCLIP"  # 指定训练器类型
    cfg.freeze()  # 冻结配置（防止后续修改）
    return cfg

# 4. 元数据生成工具（支持多文件夹扫描）
def create_category_json(root_dir: Path = TEMP_ROOT, output_file: Path = OUTPUT_JSON_PATH) -> None:
    """
    扫描根目录下所有类别子文件夹，生成BiomedCoOp所需的JSON元数据
    支持多类别文件夹（如：glioma_tumor、meningioma_tumor、normal_brain等）
    """
    if not root_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"根目录不存在：{str(root_dir)}")
    
    data_list = []
    # 遍历所有子文件夹（每个子文件夹对应一个类别）
    for category_dir in root_dir.iterdir():
        if not category_dir.is_dir():
            continue  # 跳过非目录文件
        
        category_name = category_dir.name
        logger.info(f"扫描类别文件夹：{category_name}")
        
        # 遍历类别文件夹下的所有图片文件
        for img_file in category_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                # 生成相对路径（格式：类别名/文件名，适配BiomedCoOp读取）
                relative_path = f"{category_name}/{img_file.name}"
                data_list.append([relative_path])  # 按模型要求封装为列表
    
    if not data_list:
        raise HTTPException(status_code=400, detail="根目录下无有效图片文件，请先上传")
    
    # 排序确保结果一致性
    data_list.sort()
    # 生成模型所需的JSON结构（train/val/test复用同一份数据，因推理仅用test集）
    final_json = {
        "test": data_list,
        "train": data_list,
        "val": data_list
    }
    
    # 写入JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)
    
    logger.info(f"元数据生成完成：{str(output_file)}，共包含 {len(data_list)} 张图片")

# 5. 推理参数验证工具
def validate_inference_params(model_dir: str, load_epoch: int) -> bool:
    """验证推理参数有效性（模型目录、轮次、checkpoint存在性）"""
    # 转换为绝对路径（防相对路径错误）
    abs_model_dir = os.path.abspath(model_dir)
    
    # 校验模型目录存在性
    if not os.path.isdir(abs_model_dir):
        logger.error(f"模型目录不存在：{abs_model_dir}")
        return False
    
    # 校验轮次合法性
    if load_epoch <= 0:
        logger.error(f"无效的加载轮次：{load_epoch}（必须为正整数）")
        return False
    
    # 校验checkpoint文件存在性（BiomedCoOp的prompt_learner权重路径）
    checkpoint_path = os.path.join(abs_model_dir, f"prompt_learner/model.pth.tar-{load_epoch}")
    if not os.path.isfile(checkpoint_path):
        logger.error(f"模型权重文件不存在：{checkpoint_path}")
        return False
    
    return True

# 6. 临时文件清理工具（支持多文件夹）
def clear_temp_files(root_dir: Path = TEMP_ROOT, exclude_dirs: List[str] = None) -> None:
    """
    清理根目录下所有子文件夹的图片文件（保留目录结构）
    exclude_dirs：需保留文件的目录列表（可选）
    """
    exclude_dirs = exclude_dirs or []
    logger.info(f"开始清理临时文件，保留目录：{exclude_dirs}")
    
    for category_dir in root_dir.iterdir():
        if not category_dir.is_dir():
            continue
        # 跳过需保留的目录
        if category_dir.name in exclude_dirs:
            logger.info(f"跳过保留目录：{category_dir.name}")
            continue
        
        # 删除目录下所有有效图片文件
        for file in category_dir.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                file.unlink()  # 删除文件
                logger.info(f"删除临时文件：{str(file)}")
    
    logger.info("临时文件清理完成")

# ------------------------------
# 认证依赖（统一权限控制）
# ------------------------------
async def get_current_user(
    token: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))
) -> UserInDB:
    """获取当前登录用户（所有受保护接口依赖）"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证令牌或未登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # 校验令牌存在性
    if not token:
        raise credentials_exception
    
    try:
        # 解码令牌
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError as e:
        logger.error(f"令牌解码失败：{str(e)}")
        raise credentials_exception
    
    # 校验用户存在性
    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception
    
    return user

# ------------------------------
# 服务启动事件（初始化环境）
# ------------------------------
@app.on_event("startup")
async def startup_event():
    """服务启动时初始化（创建根目录、日志记录）"""
    # 确保根目录存在
    TEMP_ROOT.mkdir(exist_ok=True, parents=True)
    # 记录启动日志
    logger.info("=== BiomedCoOp 推理服务启动完成 ===")
    logger.info(f"根目录：{str(TEMP_ROOT)}")
    logger.info(f"支持图片格式：{IMAGE_EXTENSIONS}")

# ------------------------------
# 1. 用户认证接口（保留原有功能）
# ------------------------------
@app.post("/api/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """用户注册接口"""
    # 校验用户名是否已存在
    if get_user_by_username(user.username):
        raise HTTPException(status_code=400, detail="用户名已被占用")
    
    # 创建新用户
    new_user = UserInDB(
        id=fake_db["next_id"],
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=get_password_hash(user.password),
        created_at=datetime.now(timezone.utc)
    )
    
    # 存入内存数据库
    fake_db["users"].append(new_user)
    fake_db["next_id"] += 1
    
    # 记录日志
    logger.info(f"用户注册成功：{user.username}（邮箱：{user.email}）")
    return new_user

@app.post("/api/login", response_model=TokenResponse)
async def login(login_data: LoginRequest):
    """用户登录接口（返回JWT令牌）"""
    # 校验用户存在性
    user = get_user_by_username(login_data.username)
    if not user or not verify_password(login_data.password, user.hashed_password):
        logger.warning(f"登录失败：{login_data.username}（用户名或密码错误）")
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    
    # 生成令牌
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # 记录日志
    logger.info(f"用户登录成功：{user.username}（令牌有效期：{ACCESS_TOKEN_EXPIRE_MINUTES}分钟）")
    return {
        "token": access_token,
        "user_info": user
    }

@app.get("/api/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return current_user

# ------------------------------
# 2. 文件上传接口（核心：支持单/多文件+多文件夹）
# ------------------------------
@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),  # 支持单/多文件（List接收1个或多个）
    folder: str = Query(DEFAULT_UPLOAD_FOLDER, description="目标子文件夹名称（支持多类别）"),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    单/多文件上传接口（支持多文件夹分类）
    - files：上传的图片文件（可选择1个或多个）
    - folder：目标子文件夹（如：glioma_tumor、normal_brain，自动创建）
    - 自动过滤非图片文件，校验文件大小
    """
    # 1. 初始化上传信息
    upload_info = {
        "folder": folder,
        "target_dir": str(get_upload_target_dir(folder)),
        "files": [],
        "total_count": len(files),
        "success_count": 0,
        "failed_count": 0
    }
    
    # 2. 遍历处理每个文件
    for file in files:
        file_record = {
            "original_name": file.filename,
            "safe_name": secure_filename(file.filename),
            "status": "failed",
            "reason": "",
            "size": "0KB"
        }
        
        try:
            # 读取文件内容（用于大小校验和保存）
            file_content = await file.read()
            file_size = len(file_content)
            
            # 3. 校验文件大小
            if file_size > MAX_UPLOAD_SIZE:
                file_record["reason"] = f"文件过大（最大100MB，当前{file_size/1024/1024:.2f}MB）"
                upload_info["failed_count"] += 1
                upload_info["files"].append(file_record)
                continue
            
            # 4. 校验文件格式（后缀）
            file_suffix = Path(file.filename).suffix.lower()
            if file_suffix not in IMAGE_EXTENSIONS:
                file_record["reason"] = f"不支持的文件格式（仅支持{IMAGE_EXTENSIONS}）"
                upload_info["failed_count"] += 1
                upload_info["files"].append(file_record)
                continue
            
            # 5. 保存文件到目标目录
            target_path = get_upload_target_dir(folder) / file_record["safe_name"]
            with open(target_path, "wb") as f:
                f.write(file_content)
            
            # 6. 更新文件记录（成功状态）
            file_record["status"] = "success"
            file_record["path"] = str(target_path)
            file_record["size"] = f"{file_size/1024:.2f}KB"
            upload_info["success_count"] += 1
        
        except Exception as e:
            # 捕获异常（如权限不足、磁盘满等）
            file_record["reason"] = f"服务器错误：{str(e)[:50]}"  # 截取短错误信息
            upload_info["failed_count"] += 1
        
        finally:
            # 重置文件指针（避免后续复用异常）
            await file.seek(0)
            upload_info["files"].append(file_record)
    
    # 7. 记录上传日志
    logger.info(
        f"文件上传完成：用户={current_user.username}，文件夹={folder}，"
        f"总数量={upload_info['total_count']}，成功={upload_info['success_count']}，失败={upload_info['failed_count']}"
    )
    
    # 8. 返回上传结果
    return {
        "msg": f"上传完成（成功{upload_info['success_count']}个/失败{upload_info['failed_count']}个）",
        "data": upload_info
    }

# ------------------------------
# 3. 推理接口（核心：上传后真正推理，支持多文件夹数据）
# ------------------------------
@app.post("/api/predict-demo", response_model=InferenceResponse)
async def predict_demo(
    params: InferenceParams = InferenceParams(),  # 推理参数（默认值适配大多数场景）
    current_user: UserInDB = Depends(get_current_user)
):
    """
    完整推理接口（需先上传文件）
    流程：检查上传文件 → 生成元数据 → 加载模型 → 执行推理 → 返回结果 → 清理临时文件
    """
    # 1. 前置检查：是否有上传文件
    total_files = count_uploaded_files()
    if total_files == 0:
        logger.warning(f"推理失败：用户={current_user.username}，无上传文件")
        raise HTTPException(status_code=400, detail="无有效上传文件，请先调用/api/upload接口上传图片")
    
    # 2. 验证推理参数
    if not validate_inference_params(params.model_dir, params.load_epoch):
        raise HTTPException(status_code=400, detail="推理参数无效（模型目录/轮次错误或权重缺失）")
    
    try:
        # 3. 记录推理开始时间
        start_time = time.perf_counter()
        logger.info(
            f"推理开始：用户={current_user.username}，模型目录={params.model_dir}，"
            f"加载轮次={params.load_epoch}，待推理文件数={total_files}"
        )
        
        # 4. 生成元数据JSON（扫描所有上传的类别文件夹）
        create_category_json()
        
        # 5. 加载模型配置
        cfg = setup_cfg(params.dataset_config)
        
        # 6. 初始化模型（支持CUDA加速）
        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True  # 开启卷积加速
            logger.info("使用CUDA加速推理")
        else:
            logger.info("使用CPU推理（CUDA不可用）")
        
        # 7. 构建训练器（BiomedCoOp核心逻辑）
        logger.info("开始构建BiomedCoOp训练器")
        trainer = build_trainer(cfg)
        
        # 8. 加载预训练模型权重
        trainer.load_model(params.model_dir, epoch=params.load_epoch)
        logger.info(f"模型权重加载完成：{params.model_dir}/prompt_learner/model.pth.tar-{params.load_epoch}")
        
        # 9. 执行推理（BiomedCoOp的inference方法会自动读取JSON中的test集）
        logger.info("开始执行推理")
        # 注：若trainer.inference()无返回，需修改BiomedCoOp源码让其返回预测结果
        # 此处基于output.txt日志格式，提取预测结果（实际项目需根据trainer实现调整）
        inference_result = trainer.inference()  # 假设返回包含预测详情的字典
        
        # 10. 处理推理结果（适配响应格式）
        if not isinstance(inference_result, dict):
            # 若trainer无返回，从日志或临时文件读取结果（示例逻辑）
            inference_result = {
                "total_samples": total_files,
                "inference_time": f"{time.perf_counter() - start_time:.6f}秒",
                "categories": [d.name for d in TEMP_ROOT.iterdir() if d.is_dir()],
                "message": "推理完成（结果需从BiomedCoOp日志查看，建议优化trainer.inference()返回详情）"
            }
        
        # 11. 清理临时文件（保留目录结构，方便后续上传）
        clear_temp_files(exclude_dirs=[])  # 若需保留文件，可传入exclude_dirs=["glioma_tumor"]等
        
        # 12. 记录推理完成日志
        end_time = time.perf_counter()
        logger.info(
            f"推理完成：用户={current_user.username}，耗时={end_time - start_time:.6f}秒，"
            f"结果={json.dumps(inference_result, ensure_ascii=False)[:200]}..."  # 截取日志
        )
        
        # 13. 返回推理结果
        return {
            "data": inference_result
        }
    
    except Exception as e:
        # 捕获推理异常，记录日志并返回错误
        error_msg = f"推理过程异常：{str(e)[:100]}"  # 截取短错误信息
        logger.error(f"推理失败：用户={current_user.username}，原因={error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# ------------------------------
# 4. 辅助接口（清理/健康检查）
# ------------------------------
@app.delete("/api/clear-temp")
async def clear_temp(
    exclude_dirs: List[str] = Query([], description="需保留文件的目录列表"),
    current_user: UserInDB = Depends(get_current_user)
):
    """清理临时文件接口（支持保留指定目录）"""
    try:
        clear_temp_files(exclude_dirs=exclude_dirs)
        return {
            "status": "success",
            "message": f"临时文件清理完成，保留目录：{exclude_dirs}"
        }
    except Exception as e:
        error_msg = f"清理失败：{str(e)}"
        logger.error(f"用户={current_user.username}，{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def health_check():
    """服务健康检查接口"""
    return {
        "status": "running",
        "service": "BiomedCoOp 推理服务",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tips": "1. 先调用/api/register登录获取token；2. 调用/api/upload上传文件；3. 调用/api/predict-demo推理"
    }

# ------------------------------
# 服务启动入口
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    # 启动服务（0.0.0.0允许外部访问，端口8000）
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"  # uvicorn日志级别
    )