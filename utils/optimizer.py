import sys
sys.path.append("/app/bt_ps")
from p2p_server.utils import utils

class ClientOptimizer(object):
    def update_client_weight(self, args, model, global_model):
        if args.gradient_policy == utils.FEDPROX_STRATEGY:
            for idx, param in enumerate(model.parameters()):
                param.data += (args.lr * args.proxy_mu * (param.data - global_model[idx]))
