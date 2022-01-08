import numpy as np
import time
import cv2
import os

# Initialize the lists we need to interpret the results
boxes = []
confidences = []
class_ids = []


def main():
    DARKNET_PATH = '../darknet'

    # Read labels that are used on object
    labels = open(os.path.join(DARKNET_PATH, "data", "coco.names")).read().splitlines()
    # Make random colors with a seed, such that they are the same next time
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(len(labels), 3)).tolist()

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet(os.path.join(DARKNET_PATH, "cfg", "yolov3-tiny.cfg"), "yolov3-tiny.weights")
    # Determine the output layer, now this piece is not intuitive
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    image = cv2.imread(os.path.join(DARKNET_PATH, "data", "eagle.jpg"))
    # Get the shape
    h, w = image.shape[:2]
    layer_outputs = processImage(image, net, ln)
    idxs = calculateBestBoundingBoxes(layer_outputs, h, w)
    # Ensure at least one detection exists - needed otherwise flatten will fail
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Get the box information
            x, y, w, h = boxes[i]

            # Make and add text
            text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            print(text)


def processImage(image, net, layerNames):
    # Load it as a blob and feed it to the network
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    # Get the output
    layer_outputs = net.forward(layerNames)
    end = time.time()
    return layer_outputs
  

def calculateBestBoundingBoxes(layer_outputs, h, w):
    # Loop over the layers
    for output in layer_outputs:
        # For the layer loop over all detections
        for detection in output:
            # The detection first 4 entries contains the object position and size
            scores = detection[5:]
            # Then it has detection scores - it takes the one with maximal score
            class_id = np.argmax(scores).item()
            # The maximal score is the confidence
            confidence = scores[class_id].item()

            # Ensure we have some reasonable confidence, else ignorre
            if confidence > 0.3:
                # The first four entries have the location and size (center, size)
                # It needs to be scaled up as the result is given in relative size (0.0 to 1.0)
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype(int).tolist()

                # Calculate the upper corner
                x = center_x - width//2
                y = center_y - height//2

                # Add our findings to the lists
                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Only keep the best boxes of the overlapping ones
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    return idxs

if __name__ == "__main__":
    main()
