import argparse
import logging

from datetime import datetime
from tqdm import tqdm

from inference import predict
from layers import layers
from model import get_model
from perturb import knockout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repeat", action="store", default=1)
    parser.add_argument("-l", "--level", action="store", default="synapse")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%I%M%S")
    logging.basicConfig(filename=f"logs/{timestamp}.log", filemode="a", format="%(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.DEBUG)
    logging.info(f"INFO:repeats={args.repeat}, level={args.level}")

    for layer in tqdm(layers.keys(), desc="layer"):
        for fraction in tqdm([x / 10 for x in range(0, 11)], desc="fraction", leave=False):
            for repeat in tqdm(range(args.repeat), desc="repeat", leave=False):          
                model = get_model()
                knockout(model, layers[layer], fraction, level=args.level)
                accuracy = predict(model)
                
                experiment = f"EXPT:layer={layer}, repeat={str(repeat+1)}, fraction={str(fraction)}"

                logging.info(f"{experiment}, accuracy={accuracy}")