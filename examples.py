import cv2

# Load the pre-trained model
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

# Load the image
image = cv2.imread('image.jpg')

# Perform object detection
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
detections = net.forward()

# Process the detections
for detection in detections:
    # Extract class, confidence, and bounding box coordinates
    class_id = detection[0]
    confidence = detection[1]
    box = detection[2:6] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

    # Draw bounding box and label on the image
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.putText(image, f'Class: {class_id}, Confidence: {confidence}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#=================================================================================
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Load the pipeline configuration
configs = config_util.get_configs_from_pipeline_file('pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Load the checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('checkpoint/ckpt-0').expect_partial()

# Load the label map
label_map_path = 'label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the image
image_np = np.array(Image.open('image.jpg'))

# Convert the image to tensor
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

# Run the object detection
detections = detection_model(input_tensor)

# Post-process the detections
boxes = detections['detection_boxes'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.uint32)
scores = detections['detection_scores'][0].numpy()

# Visualize the results
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=0.3,
    agnostic_mode=False
)

# Display the result
plt.imshow(image_np)
plt.show()


#======================================================================
import torch
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the image
image = Image.open('image.jpg')

# Preprocess the image
transform = transforms.Compose([
    transforms.ToTensor()
])
input_image = transform(image)
input_image = input_image.unsqueeze(0)

# Perform object detection
with torch.no_grad():
    output = model(input_image)

# Process the detections
boxes = output[0]['boxes'].tolist()
labels = output[0]['labels'].tolist()
scores = output[0]['scores'].tolist()

# Display the results
for box, label, score in zip(boxes, labels, scores):
    print(f'Class: {label}, Confidence: {score}')
    print(f'Bounding Box: {box}')

#=======================================================================
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load the configuration
cfg = get_cfg()
cfg.merge_from_file('config.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = 'model.pth'
cfg.MODEL.DEVICE = 'cuda'

# Create the predictor
predictor = DefaultPredictor(cfg)

# Load the image
image = cv2.imread('image.jpg')

# Perform object detection
outputs = predictor(image)

# Visualize the results
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
result = v.get_image()

# Display the result
cv2.imshow('Object Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
#================================================================================
import mxnet as mx
from mxnet import image
from gluoncv import model_zoo, data, utils

# Load the pre-trained model
net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)

# Load the image
image_path = 'image.jpg'
x, img = data.transforms.presets.yolo.load_test(image_path, short=512)

# Perform object detection
class_ids, scores, bounding_boxes = net(x)

# Process the detections
objects = []
for cid, score, bbox in zip(class_ids[0], scores[0], bounding_boxes[0]):
    class_name = net.classes[cid]
    confidence = float(score.asscalar())
    x_min, y_min, x_max, y_max = [int(coord.asscalar()) for coord in bbox]
    objects.append({
        'class_name': class_name,
        'confidence': confidence,
        'bbox': (x_min, y_min, x_max, y_max)
    })

# Visualize the results
ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_ids[0], class_names=net.classes)
plt.show()
