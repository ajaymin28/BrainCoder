import argparse
import torch
import os
from dataclasses import dataclass, field
from typing import List
from omegaconf import OmegaConf

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', '1'): return True
    if v.lower() in ('no', 'false', 'f', '0'): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


@dataclass
class UserLoRAConfig:
    r: int = 32
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ])
    max_seq_length: int = 2048
    #QLoRA
    load_in_4bit: bool = True
    load_in_8bit: bool = False # more precise bet takes more mem
    
class GlobalConfig:
    """
    Global configuration settings for the EEG-Image project.
    Centralizes paths and other high-level settings.
    """
    # --- Base Directories ---
    # Adjust these paths based on your project structure and environment
    PROJECT_ROOT = "D:\\Datasets\\EEG DATASET\\things2\\BrainCoder\\dump"
    DATA_BASE_DIR = "D:\\Datasets\\EEG DATASET\\things2\\NICE-EEG"
    MODEL_BASE_DIR = os.path.join(PROJECT_ROOT, "model", "grok") # Base directory for saving models and checkpoints

    # --- Specific File/Directory Paths ---
    EEG_DATA_PATH = os.path.join(DATA_BASE_DIR, "Data", "Things-EEG2", "Preprocessed_data_250Hz")
    TEST_CENTER_PATH = os.path.join(DATA_BASE_DIR, "dnn_feature")
    IMG_DATA_PATH = os.path.join(DATA_BASE_DIR, "dnn_feature") 

@dataclass
class Configuration:   
    dataset_id: str = "ariG23498/license-detection-paligemma"
    model_id: str = "google/gemma-3-4b-pt" # "unsloth/gemma-3-4b-it"
    checkpoint_id: str = "ajaymin28/Gemma3_ObjeDet"
    push_model_to_hub: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    validate_steps_freq: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 1
    max_step_to_train: int = 100 # if model converges before training one epoch, set to 0 or -1 to disable
    finetune_method: str = "lora"  # FFT | lora | qlora
    use_unsloth: bool = False
    mm_tunable_parts: List[str] = field(default_factory=lambda: ["multi_modal_projector"]) # vision_tower,language_model
    lora: UserLoRAConfig = field(default_factory=UserLoRAConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    wandb_project_name: str = "Gemma3_LoRA"

    @classmethod
    def load(cls, main_cfg_path="configs/config.yaml", lora_cfg_path="configs/lora_config.yaml"):
        base_cfg = OmegaConf.load(main_cfg_path)
        if lora_cfg_path and os.path.exists(lora_cfg_path):
            lora_cfg = OmegaConf.load(lora_cfg_path) # TODO: Merge config into one, refer to hydra config.
            base_cfg.lora = lora_cfg
        return OmegaConf.to_container(base_cfg, resolve=True)

    
    @classmethod
    def from_yaml(cls, main_cfg_path=None, lora_cfg_path=None):
        # This will only be used if YAML files are present and wanted
        if main_cfg_path and os.path.exists(main_cfg_path):
            base_cfg = OmegaConf.load(main_cfg_path)
            if lora_cfg_path and os.path.exists(lora_cfg_path):
                lora_cfg = OmegaConf.load(lora_cfg_path)
                base_cfg.lora = lora_cfg
            cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
            # Flattening/mapping nested dicts to dataclass if needed
            lora = UserLoRAConfig(**cfg_dict["lora"]) if "lora" in cfg_dict else UserLoRAConfig()
            cfg_dict.pop("lora", None)
            return cls(**cfg_dict, lora=lora)
        else:
            return cls()  # Return default config
    
    @classmethod
    def from_args(cls):
        cfg_dict = cls.load()  # Load YAML as dict
        parser = argparse.ArgumentParser()

        # Top-level args
        parser.add_argument("--dataset_id", type=str, default=cfg_dict["dataset_id"])
        parser.add_argument("--model_id", type=str, default=cfg_dict["model_id"])
        parser.add_argument("--checkpoint_id", type=str, default=cfg_dict["checkpoint_id"])
        parser.add_argument("--push_model_to_hub", type=str2bool, default=cfg_dict["push_model_to_hub"])
        parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=cfg_dict["device"])
        parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float16")
        parser.add_argument("--batch_size", type=int, default=cfg_dict["batch_size"])
        parser.add_argument("--learning_rate", type=float, default=cfg_dict["learning_rate"])
        parser.add_argument("--epochs", type=int, default=cfg_dict["epochs"])
        parser.add_argument("--max_step_to_train", type=int, default=cfg_dict["max_step_to_train"])
        parser.add_argument("--validate_steps_freq", type=int, default=cfg_dict["validate_steps_freq"])
        parser.add_argument("--finetune_method", type=str, choices=["FFT", "lora", "qlora"], default=cfg_dict["finetune_method"])
        parser.add_argument("--use_unsloth", type=str2bool, default=cfg_dict["use_unsloth"])
        parser.add_argument("--mm_tunable_parts", type=str, default=",".join(cfg_dict["mm_tunable_parts"]))

        # LoRA nested config overrides
        parser.add_argument("--lora.r", type=int, default=cfg_dict["lora"]["r"])
        parser.add_argument("--lora.alpha", type=int, default=cfg_dict["lora"]["alpha"])
        parser.add_argument("--lora.dropout", type=float, default=cfg_dict["lora"]["dropout"])
        parser.add_argument("--lora.target_modules", type=str, default=",".join(cfg_dict["lora"]["target_modules"]))
        parser.add_argument("--lora.max_seq_length", type=int, default=cfg_dict["lora"]["max_seq_length"])
        parser.add_argument("--lora.load_in_4bit", type=str2bool,default=cfg_dict["lora"]["load_in_4bit"])
        parser.add_argument("--lora.load_in_8bit", type=str2bool,default=cfg_dict["lora"]["load_in_8bit"])

        parser.add_argument("--wandb_project_name", type=str, default=cfg_dict["wandb_project_name"])

        args = parser.parse_args()

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        lora_config = UserLoRAConfig(
            r=args.__dict__["lora.r"],
            alpha=args.__dict__["lora.alpha"],
            dropout=args.__dict__["lora.dropout"],
            target_modules=[x.strip() for x in args.__dict__["lora.target_modules"].split(',')],
            max_seq_length=args.__dict__["lora.max_seq_length"],
            load_in_4bit=args.__dict__["lora.load_in_4bit"],
            load_in_8bit=args.__dict__["lora.load_in_8bit"],
        )

        # TODO handle this long list, probably migrate to hydra conf.
        return cls(
            dataset_id=args.dataset_id,
            model_id=args.model_id,
            checkpoint_id=args.checkpoint_id,
            device=args.device,
            dtype=dtype_map[args.dtype],
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            finetune_method=args.finetune_method,
            use_unsloth=args.use_unsloth,
            mm_tunable_parts=[x.strip() for x in args.mm_tunable_parts.split(',')],
            lora=lora_config,
            wandb_project_name=args.wandb_project_name,
            max_step_to_train=args.max_step_to_train,
            push_model_to_hub=args.push_model_to_hub,
            validate_steps_freq=args.validate_steps_freq
        )