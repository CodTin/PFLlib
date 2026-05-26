#!/usr/bin/env python3
"""Migration script: copy files from old structure to pfllib/ package with import rewriting."""
import os
import re
import shutil

BASE = "/Users/xiaoyu/Projects/PFLlib"

# ── File mappings: (src, dst) ──
FILE_MAPPINGS = []

# Models
FILE_MAPPINGS += [
    (f"{BASE}/system/flcore/trainmodel/models.py", f"{BASE}/pfllib/models/cnn.py"),
    (f"{BASE}/system/flcore/trainmodel/resnet.py", f"{BASE}/pfllib/models/resnet.py"),
    (f"{BASE}/system/flcore/trainmodel/alexnet.py", f"{BASE}/pfllib/models/alexnet.py"),
    (f"{BASE}/system/flcore/trainmodel/mobilenet_v2.py", f"{BASE}/pfllib/models/mobilenet.py"),
    (f"{BASE}/system/flcore/trainmodel/bilstm.py", f"{BASE}/pfllib/models/bilstm.py"),
    (f"{BASE}/system/flcore/trainmodel/transformer.py", f"{BASE}/pfllib/models/transformer.py"),
]

# Optimizers
FILE_MAPPINGS += [
    (f"{BASE}/system/flcore/optimizers/fedoptimizer.py", f"{BASE}/pfllib/optimizers/fed_optimizers.py"),
]

# Clients - mapping old names to new short names
CLIENT_MAP = {
    "clientavg": "avg",
    "clientprox": "prox",
    "clientditto": "ditto",
    "clientper": "per",
    "clientrep": "rep",
    "clientmoon": "moon",
    "clientbabu": "babu",
    "clientbn": "bn",
    "clientala": "ala",
    "clientamp": "amp",
    "clientapfl": "apfl",
    "clientapple": "apple",
    "clientas": "as",
    "clientcac": "cac",
    "clientcp": "cp",
    "clientcross": "cross",
    "clientda": "da",
    "clientdbe": "dbe",
    "clientdyn": "dyn",
    "clientfd": "fd",
    "clientfml": "fml",
    "clientfomo": "fomo",
    "clientgc": "gc",
    "clientgen": "gen",
    "clientgh": "gh",
    "clientgpfl": "gpfl",
    "clientkd": "kd",
    "clientlc": "lc",
    "clientlg": "lg",
    "clientmtl": "mtl",
    "clientntd": "ntd",
    "clientpac": "pac",
    "clientpcl": "pcl",
    "clientperavg": "peravg",
    "clientpFedMe": "pfedme",
    "clientphp": "php",
    "clientproto": "proto",
    "clientrod": "rod",
    "clientscaffold": "scaffold",
}
for old, new in CLIENT_MAP.items():
    FILE_MAPPINGS.append(
        (f"{BASE}/system/flcore/clients/{old}.py", f"{BASE}/pfllib/clients/{new}.py")
    )

# Servers
SERVER_MAP = {
    "serveravg": "avg",
    "serverlocal": "local",
    "serverprox": "prox",
    "serverditto": "ditto",
    "serverper": "per",
    "serverrep": "rep",
    "servermoon": "moon",
    "serverbabu": "babu",
    "serverbn": "bn",
    "serverala": "ala",
    "serveramp": "amp",
    "serverapfl": "apfl",
    "serverapple": "apple",
    "serveras": "as",
    "servercac": "cac",
    "servercp": "cp",
    "servercross": "cross",
    "serverda": "da",
    "serverdbe": "dbe",
    "serverdyn": "dyn",
    "serverfd": "fd",
    "serverfml": "fml",
    "serverfomo": "fomo",
    "servergc": "gc",
    "servergen": "gen",
    "servergh": "gh",
    "servergpfl": "gpfl",
    "serverkd": "kd",
    "serverlc": "lc",
    "serverlg": "lg",
    "servermtl": "mtl",
    "serverntd": "ntd",
    "serverpac": "pac",
    "serverpcl": "pcl",
    "serverperavg": "peravg",
    "serverpFedMe": "pfedme",
    "serverphp": "php",
    "serverproto": "proto",
    "serverrod": "rod",
    "serverscaffold": "scaffold",
}
for old, new in SERVER_MAP.items():
    FILE_MAPPINGS.append(
        (f"{BASE}/system/flcore/servers/{old}.py", f"{BASE}/pfllib/servers/{new}.py")
    )

# Data utils and generators
FILE_MAPPINGS += [
    (f"{BASE}/dataset/utils/dataset_utils.py", f"{BASE}/pfllib/data/utils.py"),
    (f"{BASE}/dataset/utils/language_utils.py", f"{BASE}/pfllib/data/language_utils.py"),
    (f"{BASE}/dataset/utils/HAR_utils.py", f"{BASE}/pfllib/data/har_utils.py"),
    (f"{BASE}/system/utils/data_utils.py", f"{BASE}/pfllib/data/reader.py"),
]

# Data generators
GEN_MAP = {
    "generate_MNIST": "mnist",
    "generate_Cifar10": "cifar10",
    "generate_Cifar100": "cifar100",
    "generate_EMNIST": "emnist",
    "generate_FashionMNIST": "fashionmnist",
    "generate_FEMNIST": "femnist",
    "generate_TinyImagenet": "tinyimagenet",
    "generate_AGNews": "agnews",
    "generate_SogouNews": "sogounews",
    "generate_Shakespeare": "shakespeare",
    "generate_AmazonReview": "amazonreview",
    "generate_Digit5": "digit5",
    "generate_DomainNet": "domainnet",
    "generate_Camelyon17": "camelyon17",
    "generate_iWildCam": "iwildcam",
    "generate_Omniglot": "omniglot",
    "generate_HAR": "har",
    "generate_PAMAP2": "pamap2",
    "generate_Country211": "country211",
    "generate_Flowers102": "flowers102",
    "generate_GTSRB": "gtsrb",
    "generate_StanfordCars": "stanfordcars",
    "generate_COVIDx": "covidx",
    "generate_kvasir": "kvasir",
}
for old, new in GEN_MAP.items():
    FILE_MAPPINGS.append(
        (f"{BASE}/dataset/{old}.py", f"{BASE}/pfllib/data/generators/{new}.py")
    )

# Utils and privacy
FILE_MAPPINGS += [
    (f"{BASE}/system/utils/ALA.py", f"{BASE}/pfllib/utils/ala.py"),
    (f"{BASE}/system/utils/mem_utils.py", f"{BASE}/pfllib/utils/mem.py"),
    (f"{BASE}/system/utils/result_utils.py", f"{BASE}/pfllib/utils/results.py"),
    (f"{BASE}/system/utils/dlg.py", f"{BASE}/pfllib/privacy/dlg.py"),
]


def rewrite_imports(content: str, dst: str) -> str:
    """Rewrite old import paths to new pfllib paths."""
    
    # Common import rewrites for all files
    replacements = [
        # Client imports
        (r'from flcore\.clients\.clientbase import Client',
         'from pfllib.clients.base import Client'),
        (r'from flcore\.clients\.clientavg import clientAVG',
         'from pfllib.clients.avg import ClientAVG'),
        (r'from flcore\.clients\.clientprox import clientProx',
         'from pfllib.clients.prox import ClientProx'),
        (r'from flcore\.clients\.clientditto import clientDitto',
         'from pfllib.clients.ditto import ClientDitto'),
        (r'from flcore\.clients\.clientper import clientPer',
         'from pfllib.clients.per import ClientPer'),
        (r'from flcore\.clients\.clientrep import clientRep',
         'from pfllib.clients.rep import ClientRep'),
        (r'from flcore\.clients\.clientmoon import clientMOON',
         'from pfllib.clients.moon import ClientMOON'),
        (r'from flcore\.clients\.clientbabu import clientBABU',
         'from pfllib.clients.babu import ClientBABU'),
        (r'from flcore\.clients\.clientbn import clientBN',
         'from pfllib.clients.bn import ClientBN'),
        (r'from flcore\.clients\.clientala import clientALA',
         'from pfllib.clients.ala import ClientALA'),
        (r'from flcore\.clients\.clientamp import clientAMP',
         'from pfllib.clients.amp import ClientAMP'),
        (r'from flcore\.clients\.clientapfl import clientAPFL',
         'from pfllib.clients.apfl import ClientAPFL'),
        (r'from flcore\.clients\.clientapple import clientAPPLE',
         'from pfllib.clients.apple import ClientAPPLE'),
        (r'from flcore\.clients\.clientas import clientAS',
         'from pfllib.clients.as import ClientAS'),
        (r'from flcore\.clients\.clientcac import clientCAC',
         'from pfllib.clients.cac import ClientCAC'),
        (r'from flcore\.clients\.clientcp import clientCP',
         'from pfllib.clients.cp import ClientCP'),
        (r'from flcore\.clients\.clientcp import \*',
         'from pfllib.clients.cp import *'),
        (r'from flcore\.clients\.clientcross import clientCross',
         'from pfllib.clients.cross import ClientCross'),
        (r'from flcore\.clients\.clientda import clientDA',
         'from pfllib.clients.da import ClientDA'),
        (r'from flcore\.clients\.clientdbe import clientDBE',
         'from pfllib.clients.dbe import ClientDBE'),
        (r'from flcore\.clients\.clientdyn import clientDyn',
         'from pfllib.clients.dyn import ClientDyn'),
        (r'from flcore\.clients\.clientfd import clientFD',
         'from pfllib.clients.fd import ClientFD'),
        (r'from flcore\.clients\.clientfml import clientFML',
         'from pfllib.clients.fml import ClientFML'),
        (r'from flcore\.clients\.clientfomo import clientFomo',
         'from pfllib.clients.fomo import ClientFomo'),
        (r'from flcore\.clients\.clientgc import clientGC',
         'from pfllib.clients.gc import ClientGC'),
        (r'from flcore\.clients\.clientgen import clientGen',
         'from pfllib.clients.gen import ClientGen'),
        (r'from flcore\.clients\.clientgh import clientGH',
         'from pfllib.clients.gh import ClientGH'),
        (r'from flcore\.clients\.clientgpfl import clientGPFL',
         'from pfllib.clients.gpfl import ClientGPFL'),
        (r'from flcore\.clients\.clientkd import clientKD',
         'from pfllib.clients.kd import ClientKD'),
        (r'from flcore\.clients\.clientlc import clientLC',
         'from pfllib.clients.lc import ClientLC'),
        (r'from flcore\.clients\.clientlg import clientLG',
         'from pfllib.clients.lg import ClientLG'),
        (r'from flcore\.clients\.clientmtl import clientMTL',
         'from pfllib.clients.mtl import ClientMTL'),
        (r'from flcore\.clients\.clientntd import clientNTD',
         'from pfllib.clients.ntd import ClientNTD'),
        (r'from flcore\.clients\.clientpac import clientPAC',
         'from pfllib.clients.pac import ClientPAC'),
        (r'from flcore\.clients\.clientpcl import clientPCL',
         'from pfllib.clients.pcl import ClientPCL'),
        (r'from flcore\.clients\.clientperavg import clientPerAvg',
         'from pfllib.clients.peravg import ClientPerAvg'),
        (r'from flcore\.clients\.clientpFedMe import clientpFedMe',
         'from pfllib.clients.pfedme import ClientPFedMe'),
        (r'from flcore\.clients\.clientphp import clientPHP',
         'from pfllib.clients.php import ClientPHP'),
        (r'from flcore\.clients\.clientproto import clientProto',
         'from pfllib.clients.proto import ClientProto'),
        (r'from flcore\.clients\.clientrod import clientROD',
         'from pfllib.clients.rod import ClientROD'),
        (r'from flcore\.clients\.clientscaffold import clientScaffold',
         'from pfllib.clients.scaffold import ClientScaffold'),
        
        # Server imports
        (r'from flcore\.servers\.serverbase import Server',
         'from pfllib.servers.base import Server'),
        
        # Optimizer imports
        (r'from flcore\.optimizers\.fedoptimizer import',
         'from pfllib.optimizers.fed_optimizers import'),
        
        # Model imports - wildcard
        (r'from flcore\.trainmodel\.models import \*',
         'from pfllib.models.cnn import *'),
        (r'from flcore\.trainmodel\.bilstm import \*',
         'from pfllib.models.bilstm import *'),
        (r'from flcore\.trainmodel\.resnet import \*',
         'from pfllib.models.resnet import *'),
        (r'from flcore\.trainmodel\.alexnet import \*',
         'from pfllib.models.alexnet import *'),
        (r'from flcore\.trainmodel\.mobilenet_v2 import \*',
         'from pfllib.models.mobilenet import *'),
        (r'from flcore\.trainmodel\.transformer import \*',
         'from pfllib.models.transformer import *'),
        
        # Utils imports
        (r'from utils\.data_utils import read_client_data',
         'from pfllib.data.reader import read_client_data'),
        (r'from utils\.dlg import DLG',
         'from pfllib.privacy.dlg import DLG'),
        (r'from utils\.ALA import ALA',
         'from pfllib.utils.ala import ALA'),
        (r'from utils\.mem_utils import MemReporter',
         'from pfllib.utils.mem import MemReporter'),
        (r'from utils\.result_utils import average_data',
         'from pfllib.utils.results import average_data'),
        
        # Dataset utils imports (for generators)
        (r'from utils\.dataset_utils import',
         'from pfllib.data.utils import'),
        (r'from utils\.language_utils import',
         'from pfllib.data.language_utils import'),
        (r'from utils\.HAR_utils import \*',
         'from pfllib.data.har_utils import *'),
        (r'from utils\.HAR_utils import',
         'from pfllib.data.har_utils import'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # File-specific rewrites
    if 'pfllib/data/reader.py' in dst:
        content = content.replace(
            "os.path.join('../dataset', dataset, 'train/')",
            "str(config_or_data_dir / dataset / 'train/') if hasattr(config_or_data_dir, '__truediv__') else os.path.join(str(config_or_data_dir), dataset, 'train/')"
        )
        content = content.replace(
            "os.path.join('../dataset', dataset, 'test/')",
            "str(config_or_data_dir / dataset / 'test/') if hasattr(config_or_data_dir, '__truediv__') else os.path.join(str(config_or_data_dir), dataset, 'test/')"
        )
    
    if 'pfllib/data/generators/' in dst:
        content = content.replace('from utils.', 'from pfllib.data.')
        content = content.replace('from utils ', 'from pfllib.data.utils ')
    
    return content


def migrate_file(src: str, dst: str) -> None:
    """Read source, rewrite imports, write to destination."""
    if not os.path.exists(src):
        print(f"  SKIP (not found): {src}")
        return
    
    with open(src, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = rewrite_imports(content, dst)
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  OK: {os.path.basename(src)} -> {os.path.relpath(dst, BASE)}")


def main():
    print("Starting migration...")
    for src, dst in FILE_MAPPINGS:
        migrate_file(src, dst)
    print(f"\nMigrated {len(FILE_MAPPINGS)} files.")


if __name__ == "__main__":
    main()
