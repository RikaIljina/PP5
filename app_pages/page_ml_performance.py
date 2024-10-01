import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation, load_reports


def page_ml_performance_metrics():

    st.write("### Train, Validation and Test Set: Labels Frequencies")
    labels_distribution = plt.imread(f"outputs/labels_distribution_after_split.png")
    buff, col1, buff = st.columns([2, 6, 2])
    col1.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")

    st.write("### Train, Validation and Test Set: Balancing and Augmentation")

    augmented_train_set = plt.imread(f"outputs/post_augment_montage_train.png")
    st.image(augmented_train_set, caption='Train set after augmentation')
    st.write(f"Train set preparation and augmentation:\n\n"
             f"* noise, cropping ...")

    st.write("### Model Setup")
    eval, hyperparams = load_test_evaluation()
    model_df = pd.DataFrame.from_dict(hyperparams, orient='index')
    model_df.rename(columns={0: 'Value'}, inplace=True)
    st.write("Model type, activation functions, layers...")
    st.dataframe(model_df)
    

    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/model_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/model_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(eval, index=['Loss', 'Accuracy'], columns=['Value']))
    
    st.write("### Detailed Performance on Test Set")
    
    test_report = load_reports()['test']
    st.code(f'''{test_report}''' )
    
    #st.dataframe(pd.DataFrame(load_reports()['test'])) 
    # fix
    # st.write(load_reports()['test'])
    
    test_heat_precision = plt.imread(f"outputs/pred_test_precision_heatmap.png")
    st.image(test_heat_precision, caption='Prediction of test set, precision')

    test_heat_classification = plt.imread(f"outputs/pred_test_classification_heatmap.png")
    st.image(test_heat_classification, caption='Prediction of test set, classification')

    
    st.write("### Detailed Performance on Live Set")

    live_heat_precision = plt.imread(f"outputs/pred_live_precision_heatmap.png")
    st.image(live_heat_precision, caption='Prediction of Live set, precision')
    
    live_heat_classification = plt.imread(f"outputs/pred_live_classification_heatmap.png")
    st.image(live_heat_classification, caption='Prediction of Live set, classification')


    st.write("... Live report, Test report as tables ...")
    
    live_report = load_reports()['live']
    st.code(f'''{live_report}''' )
    
    live_batch_report = load_reports()['live_batches']
    st.code(f'''{live_batch_report}''' )
    
    #st.dataframe(pd.DataFrame(load_reports()['live']))
    #st.markdown(f'''{pd.DataFrame(load_reports()['live'])}''')
    #st.markdown(f'''{load_reports()['live']}''')
    
    
    st.write("#### Detailed Performance on Live Batches")
    
    live_img_fin = plt.imread(f"outputs/live_class_img_probas_fin.png")
    st.image(live_img_fin, caption='Label "fin", probability spread between images')
    live_batch_fin = plt.imread(f"outputs/live_class_batch_probas_fin.png")
    st.image(live_batch_fin, caption='Label "fin", probability spread between batches')
    
    live_img_iris = plt.imread(f"outputs/live_class_img_probas_iris.png")
    st.image(live_img_iris, caption='Label "iris", probability spread between images')
    live_batch_iris = plt.imread(f"outputs/live_class_batch_probas_iris.png")
    st.image(live_batch_iris, caption='Label "iris", probability spread between batches')
    
    live_img_smilla = plt.imread(f"outputs/live_class_img_probas_smilla.png")
    st.image(live_img_smilla, caption='Label "smilla", probability spread between images')
    live_batch_smilla = plt.imread(f"outputs/live_class_batch_probas_smilla.png")
    st.image(live_batch_smilla, caption='Label "smilla", probability spread between batches')

    
    
    #st.dataframe(pd.DataFrame(load_reports()['live_batch']))
    
    
    