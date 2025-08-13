import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, VisionEncoderDecoderModel, ViTImageProcessor, \
    AutoTokenizer


def generate_caption(image, caption_model, processor, tokenizer, device):
    # Preprocess the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate caption (using beam search)
    output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4,
                                        return_dict_in_generate=True).sequences
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption


def detect_objects(image, detection_model, detection_processor, device):
    # Preprocess the image for object detection
    inputs = detection_processor(images=image, return_tensors="pt").to(device)

    # Perform object detection
    outputs = detection_model(**inputs)

    # Get detected objects
    target_sizes = torch.tensor([image.shape[:2]])
    results = detection_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    return results


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize the object detection model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    detection_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Initialize the image captioning model
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Create a resizable window
    cv2.namedWindow('Object Detection and Image Captioning', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects
        detections = detect_objects(rgb_frame, detection_model, detection_processor, device)

        # Draw bounding boxes and labels for detected objects
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            box = [int(i) for i in box.tolist()]
            color = (255, 0, 0)  # Default blue color for objects
            if label == 1:  # Assuming '1' is the label for humans in COCO dataset
                color = (0, 0, 255)  # Red color for humans
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Generate and display caption
        caption = generate_caption(rgb_frame, caption_model, caption_processor, caption_tokenizer, device)
        cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)  # Green color for captions

        # Display the output
        cv2.imshow('Object Detection and Image Captioning', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
