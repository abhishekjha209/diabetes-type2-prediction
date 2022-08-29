import gradio as gr
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
print(classifier)


def weightGraph(loaded_model):
        fig = plt.figure()
        importance = classifier.feature_importances_
        print(importance)
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
        return fig

def graphPlot2(inp):
        fig = plt.figure()
        historyOfPatient = multiline(inp)[0]
        historyOfPatient = list(map(int, historyOfPatient))
        y = historyOfPatient
        
        listOfMonths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x = listOfMonths[:len(historyOfPatient)]
        plt.plot(x, y)
        plt.xlabel('Track record as per months')
        plt.ylabel('If diabetic or not')
        plt.title('Diabetic History of you!')
        return fig

def multiline(textData):
        print("inp", textData)
        empty_array = []
        for line in textData.split("\n"):
            abc = list(line.split(","))
            empty_array.append(abc)        
        print(empty_array)
        return empty_array


def predict2(content):
        multiple_records = multiline(content)
        persistant_results = []
        for item in multiple_records:
                pregnancies = item[0]
                glucose = item[1]
                bp = item[2]
                skin_thickness = item[3]
                insulin = item[4]
                body_mass_index = item[5]
                diabetes_pedigree = item[6]
                age = item[7]
                data = np.array([[int(pregnancies), int(glucose), int(bp), int(skin_thickness), int(insulin), np.float(body_mass_index), np.float(diabetes_pedigree), int(age)]])
                result = classifier.predict(data)
                print("Result : ", result)
                if result==[0]:
                        persistant_results.append(0)
                else:
                        persistant_results.append(1)
        return persistant_results

with gr.Blocks() as demo:
        gr.Markdown("Flip text or image files using this demo.")
        with gr.Tabs():
                with gr.TabItem("Yearly Diabetic Record"):
                    text_input = gr.Textbox()
                    text_output = gr.Textbox()
                    text_button = gr.Button("Get Yearly Record")
                    #gbutton = gr.Button("")

                    graphOut = gr.Plot()
                    text_button.click(weightGraph, inputs=text_input, outputs=graphOut)
                with gr.TabItem("Diabetic History Analysis"):
                    text_input2 = gr.Textbox()
                    text_output2 = gr.Plot()
                    text_button2 = gr.Button("Get History!")
                
                
                text_button.click(predict2, inputs=text_input, outputs=text_output)
                text_button2.click(graphPlot2, inputs=text_input2, outputs=text_output2)
        
demo.launch()
