import torch
import pandas as pd
from torch.utils.data import SequentialSampler
import src.Models.Adjusted_models as models
import os
from src.data.DataObj import FaceKeypointDataSet
from src.utilities.logger import set_logger
from src.utilities.utility import text_img_to_numpy
from src.data.make_dataset import extract_file_from_raw

main_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_directory = os.path.join(main_directory, 'data')

extract_file_from_raw()
model = models.resnet50
model_name = model.__name__
model_dir = os.path.join(main_directory, 'Models', model_name, model_name + "_model.pt")
logger = set_logger(os.path.join(main_directory, 'src', 'Models', 'log', model_name + "_predict.log"))
checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.cpu()
logger.info('Model is ready!')

test_csv = pd.read_csv(os.path.join(data_directory, 'interim', 'test.csv'))
text_img_to_numpy(test_csv)
test_data = FaceKeypointDataSet(test_csv, is_test_set=True)
test_sampler = SequentialSampler(range(len(test_data)))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), sampler=test_sampler)
images = next(iter(test_loader))
with torch.no_grad():
    model.eval()
    predicted_labels = model(images)

keypts_labels = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
                 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
                 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
                 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
                 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y',
                 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
                 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'Image']
id_lookup = pd.read_csv(os.path.join(data_directory, 'raw', 'IdLookupTable.csv'))
id_lookup_features = list(id_lookup['FeatureName'])
id_lookup_image = list(id_lookup['ImageId'])

for i in range(len(id_lookup_features)):
    id_lookup_features[i] = keypts_labels.index(id_lookup_features[i])

location = []
for i in range(len(id_lookup_features)):
    value = float(predicted_labels[id_lookup_image[i] - 1][id_lookup_features[i]])
    location.append(value)
id_lookup['Location'] = location
submission = id_lookup[['RowId', 'Location']]
submission.to_csv(os.path.join(main_directory, 'submission.csv'), index=False)
logger.info('Total test images labeled:{}'.format(len(submission) - 1))
logger.info('Submission file is ready')
