from inference import predict
from model import get_net

model = get_net()

baseline_accuracy = predict(model)