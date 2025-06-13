import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultiDomainImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        domains=None,                # List of domains, or None for all
        classes=None,                # List of classes to include, or None for all
        transform=None
    ):
        self.samples = []
        self.transform = transform or transforms.ToTensor()
        self.class_to_idx_per_domain = {}
        
        # If domains not specified, discover from directory
        if domains is None:
            domains = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if isinstance(domains, list):
            domains = domains
        self.domain_to_idx = {d: i for i, d in enumerate(domains)}
        
        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            if not os.path.isdir(domain_dir):
                continue

            # Filter classes if provided
            domain_classes = sorted([
                d for d in os.listdir(domain_dir)
                if os.path.isdir(os.path.join(domain_dir, d)) and (classes is None or d in classes)
            ])
            class_to_idx = {cls: idx for idx, cls in enumerate(domain_classes)}
            self.class_to_idx_per_domain[domain] = class_to_idx

            for cls in domain_classes:
                cls_dir = os.path.join(domain_dir, cls)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(cls_dir, fname)
                        self.samples.append({
                            "img_path": img_path,
                            "class_idx": class_to_idx[cls],
                            "domain_idx": self.domain_to_idx[domain],
                            "domain": domain,
                            "class_name": cls
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["img_path"]).convert('RGB')
        img = self.transform(img)
        return {
            "image": img,
            "label": item["class_idx"],
            "domain": item["domain_idx"],
            # "domain_name": item["domain"],
            # "class_name": item["class_name"]
        }
