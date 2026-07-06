import os


class Initialization:
    def __init__(self, config):
        self.config = config
        self.init_from = 'scratch'

    def check_finished(self):
        """
        mkdir -p "${OUTPUT_DIR}"
        if [ -f "${OUTPUT_DIR}/finished" ]; then echo "${RUN_NAME} already finished"; exit 0; fi
        if [ -f "${OUTPUT_DIR}/checkpoints/ckpt.pt" ]; then INIT=resume; else INIT=scratch; fi
        """
        out_dir = self.config['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(os.path.join(out_dir, 'finished')):
            print(f"{self.config.get('wandb_run_name', out_dir)} already finished")
            return False
        self.init_from = 'resume' if os.path.exists(
            os.path.join(out_dir, 'checkpoints', 'ckpt.pt')) else 'scratch'
        return True

    def begin(self):
        if not self.check_finished():
            return False
        return True
