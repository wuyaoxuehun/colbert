from omegaconf import OmegaConf
from proj_conf.training_arguments import MyTraniningArgs
from transformers import TrainingArguments

# import tempfile
# tempfile.NamedTemporaryFile()
def omega_test():
    args = MyTraniningArgs(output_dir="./temp")
    print(args)
    conf = OmegaConf.create(args)
    # conf = OmegaConf.structured(TrainingArguments(output_dir="./temp"))
    print(conf)
    print(conf.logging_steps)

if __name__ == '__main__':
    omega_test()

