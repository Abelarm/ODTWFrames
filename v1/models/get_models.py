
def get_model_function(dataset_type, network_type, appendix):

    if dataset_type == 'TS':

        if network_type == '1DCNN':
            from models.CNN1D.model import get_model, optimizer
        elif network_type == 'ResNet':
            from models.ResNet1D.model import get_model, optimizer
        elif network_type == 'RNN':
            from models.RNN.model import get_model, optimizer

    elif dataset_type == 'DTW':

        if network_type == 'CNN':

            if appendix == '100k':
                from models.CNN.model_100k import get_model, optimizer
            elif appendix == '1M':
                from models.CNN.model_1M import get_model, optimizer
            else:
                from models.CNN.model import get_model, optimizer

        if network_type == 'ResNet':

            if appendix == '100k':
                from models.ResNet.model_100k import get_model, optimizer
            elif appendix == '1M':
                from models.ResNet.model_1M import get_model, optimizer
            else:
                from models.ResNet.model import get_model, optimizer

    return get_model, optimizer
