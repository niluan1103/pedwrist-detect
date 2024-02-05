from ultralytics import YOLO
from PIL import Image

# Reference: https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8

model = YOLO('GRAZPEDWRI-DX_YOLOv8_best.pt')

# Inference with best weight
def wrist_predict(PIL_im):
    #PIL_im is PIL.Image object
    results = model.predict(PIL_im, conf=0.25)
    return results

def results_to_img(results):
    # Save the results to images file
    i = 0
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('data/result_'+str(i)+'.jpg')  # save image
        i+=1

def get_results_count(results):
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        confidence_scores = boxes.conf.tolist()
        class_values = boxes.cls.tolist()
        class_values = [int(val) for val in class_values]
        #print(result.names)
        class_count = {}
        for key,name in result.names.items():
            i=0
            for class_value in class_values:
                if int(class_value) == int(key):
                    i+=1
                    class_count[key] = [name,i]

    print(confidence_scores)
    print(class_values)
    #print(class_count)
    #class_count ={class_value:[class_name,found_number]} <-dict
    for x,y in class_count.items():
        #y = [class_name,found_number]
        if y[1] ==1:
            print('Found ' + str(y[1]) + ' ' + y[0])
        else:
            print('Found ' + str(y[1]) + ' ' + y[0] + 's')
    #print(xyxys)
    #print(boxes)
    return class_count

def get_results_text(results_count):
    #results_count ={class_value:[class_name,found_number]} <-dict
    if len(results_count) == 0:
        result_text = "No abnomality is found"
    else:
        result_text = "Found"
        for x,y in results_count.items():
            if (y[0] != "text"):
                if y[1] == 1:
                    result_text = result_text + ' ' + str(y[1]) + ' ' + y[0] + ','
                else:
                    result_text = result_text + ' ' + str(y[1]) + ' ' + y[0] + 's,'
        #remove last character (",")
        result_text = result_text.rstrip(result_text[-1])
    return result_text


if __name__ == '__main__':
    img_path = 'data/default.jpg'
    img = Image.open(img_path)
    predict_results = wrist_predict(img)
    results_to_img(predict_results)
    results_count = get_results_count(predict_results)
    result_text = get_results_text(results_count)
    print(result_text)