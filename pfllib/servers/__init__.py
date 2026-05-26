from pfllib.clients import (
    ClientALA,
    ClientAMP,
    ClientAPFL,
    ClientAPPLE,
    ClientAS,
    ClientAVG,
    ClientBABU,
    ClientBN,
    ClientCAC,
    ClientCP,
    ClientCross,
    ClientDA,
    ClientDBE,
    ClientDitto,
    ClientDyn,
    ClientFD,
    ClientFML,
    ClientFomo,
    ClientGC,
    ClientGen,
    ClientGH,
    ClientGPFL,
    ClientKD,
    ClientLC,
    ClientLG,
    ClientMOON,
    ClientMTL,
    ClientNTD,
    ClientPAC,
    ClientPCL,
    ClientPer,
    ClientPerAvg,
    ClientpFedMe,
    ClientPHP,
    ClientProto,
    ClientProx,
    ClientRep,
    ClientROD,
    ClientScaffold,
)
from pfllib.registry import register_algorithm
from pfllib.servers.ala import FedALA
from pfllib.servers.amp import FedAMP
from pfllib.servers.apfl import APFL
from pfllib.servers.apple import APPLE
from pfllib.servers.asfl import FedAS
from pfllib.servers.avg import FedAvg
from pfllib.servers.babu import FedBABU
from pfllib.servers.base import Server
from pfllib.servers.bn import FedBN
from pfllib.servers.cac import FedCAC
from pfllib.servers.cp import FedCP
from pfllib.servers.cross import FedCross
from pfllib.servers.da import PFL_DA
from pfllib.servers.dbe import FedDBE
from pfllib.servers.ditto import Ditto
from pfllib.servers.dyn import FedDyn
from pfllib.servers.fd import FD
from pfllib.servers.fml import FML
from pfllib.servers.fomo import FedFomo
from pfllib.servers.gc import FedGC
from pfllib.servers.gen import FedGen
from pfllib.servers.gh import FedGH
from pfllib.servers.gpfl import GPFL
from pfllib.servers.kd import FedKD
from pfllib.servers.lc import FedLC
from pfllib.servers.lg import LG_FedAvg
from pfllib.servers.local import Local
from pfllib.servers.moon import MOON
from pfllib.servers.mtl import FedMTL
from pfllib.servers.ntd import FedNTD
from pfllib.servers.pac import FedPAC
from pfllib.servers.pcl import FedPCL
from pfllib.servers.per import FedPer
from pfllib.servers.peravg import PerAvg
from pfllib.servers.pfedme import pFedMe
from pfllib.servers.php import FedPHP
from pfllib.servers.proto import FedProto
from pfllib.servers.prox import FedProx
from pfllib.servers.rep import FedRep
from pfllib.servers.rod import FedROD
from pfllib.servers.scaffold import SCAFFOLD

register_algorithm("FedAvg", client_cls=ClientAVG, uses_head_split=True)(FedAvg)
register_algorithm("Local", client_cls=ClientAVG, uses_head_split=False)(Local)
register_algorithm("FedProx", client_cls=ClientProx, uses_head_split=False)(FedProx)
register_algorithm("Ditto", client_cls=ClientDitto, uses_head_split=False)(Ditto)
register_algorithm("FedPer", client_cls=ClientPer, uses_head_split=True)(FedPer)
register_algorithm("FedRep", client_cls=ClientRep, uses_head_split=True)(FedRep)
register_algorithm("MOON", client_cls=ClientMOON, uses_head_split=True)(MOON)
register_algorithm("FedBABU", client_cls=ClientBABU, uses_head_split=True)(FedBABU)
register_algorithm("FedBN", client_cls=ClientBN, uses_head_split=False)(FedBN)
register_algorithm("FedALA", client_cls=ClientALA, uses_head_split=False)(FedALA)
register_algorithm("FedAMP", client_cls=ClientAMP, uses_head_split=False)(FedAMP)
register_algorithm("APFL", client_cls=ClientAPFL, uses_head_split=False)(APFL)
register_algorithm("APPLE", client_cls=ClientAPPLE, uses_head_split=False)(APPLE)
register_algorithm("FedAS", client_cls=ClientAS, uses_head_split=True)(FedAS)
register_algorithm("FedCAC", client_cls=ClientCAC, uses_head_split=False)(FedCAC)
register_algorithm("FedCP", client_cls=ClientCP, uses_head_split=True)(FedCP)
register_algorithm("FedCross", client_cls=ClientCross, uses_head_split=False)(FedCross)
register_algorithm("PFL-DA", client_cls=ClientDA, uses_head_split=True)(PFL_DA)
register_algorithm("FedDBE", client_cls=ClientDBE, uses_head_split=True)(FedDBE)
register_algorithm("FedDyn", client_cls=ClientDyn, uses_head_split=False)(FedDyn)
register_algorithm("FD", client_cls=ClientFD, uses_head_split=False)(FD)
register_algorithm("FML", client_cls=ClientFML, uses_head_split=False)(FML)
register_algorithm("FedFomo", client_cls=ClientFomo, uses_head_split=False)(FedFomo)
register_algorithm("FedGC", client_cls=ClientGC, uses_head_split=True)(FedGC)
register_algorithm("FedGen", client_cls=ClientGen, uses_head_split=True)(FedGen)
register_algorithm("FedGH", client_cls=ClientGH, uses_head_split=True)(FedGH)
register_algorithm("GPFL", client_cls=ClientGPFL, uses_head_split=True)(GPFL)
register_algorithm("FedKD", client_cls=ClientKD, uses_head_split=True)(FedKD)
register_algorithm("FedLC", client_cls=ClientLC, uses_head_split=True)(FedLC)
register_algorithm("LG-FedAvg", client_cls=ClientLG, uses_head_split=True)(LG_FedAvg)
register_algorithm("FedMTL", client_cls=ClientMTL, uses_head_split=False)(FedMTL)
register_algorithm("FedNTD", client_cls=ClientNTD, uses_head_split=False)(FedNTD)
register_algorithm("FedPAC", client_cls=ClientPAC, uses_head_split=True)(FedPAC)
register_algorithm("FedPCL", client_cls=ClientPCL, uses_head_split=False)(FedPCL)
register_algorithm("PerAvg", client_cls=ClientPerAvg, uses_head_split=False)(PerAvg)
register_algorithm("pFedMe", client_cls=ClientpFedMe, uses_head_split=False)(pFedMe)
register_algorithm("FedPHP", client_cls=ClientPHP, uses_head_split=True)(FedPHP)
register_algorithm("FedProto", client_cls=ClientProto, uses_head_split=True)(FedProto)
register_algorithm("FedROD", client_cls=ClientROD, uses_head_split=True)(FedROD)
register_algorithm("SCAFFOLD", client_cls=ClientScaffold, uses_head_split=False)(SCAFFOLD)

__all__ = ["Server"]
