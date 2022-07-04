import logging
import os 


do_train = True
output_dir = "/Users/aziiz/Documents/Works/NLP/mixspeech/output"

log_filename = "{}log.log".format("" if do_train else "eval_")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(output_dir, log_filename)),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(output_dir)



