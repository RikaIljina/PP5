import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation, load_reports


def page_ml_performance_metrics():

    st.write("### Dataset preparation")

    labels_distribution = plt.imread(
        f"outputs/labels_distribution_after_split.png")
    col1, col2, = st.columns(2)
    col1.image(labels_distribution,
               caption='Initial label distribution on Train, Validation and Test Sets')
    st.write("---")

    st.write("### Train, Validation and Test Set: Balancing and Augmentation")

    augmented_dataset = plt.imread(f"outputs/post_augment_dataset_plot.png")
    col2.image(augmented_dataset,
               caption='Dataset after augmentation and balancing')

    augmented_train_set = plt.imread(f"outputs/post_augment_montage_train.png")
    st.image(augmented_train_set, caption='Train set after augmentation')
    st.write("""
            The dataset was prepared and augmented by:\n\n
            - adding gaussian noise\n
            - random cropping\n
            - random brightness adjustment\n
            - random hue adjustment\n
            The only augmentation method performed on the separately processed
            test and validation sets was random cropping.
            """)

    st.write("### Model Setup")
    st.write(
        """
        The model is a Deep Learning Convolutional Neural Network (CNN) built 
        with Keras, a high-level API for Tensorflow. The CNN model uses the 
        Softmax regression as the output layer activation function and the 
        default ReLU activation function for all other layers.\n
        Categorical cross-entropy, which is suitable for multi-class 
        classification, was used as the loss function, while Adam, being an 
        efficient and resource-friendly algorithm, was used as the optimizer.
        """)

    st.markdown(
        """         
            [Input shape: 128, 128, 3]<br>
            [Model type :  Sequential]<br>
            <table>
                <tr>
                    <th>Layers</th>
                    <th>Content</th>
                </tr>
                <tr>
                    <td>InputLayer</td>
                    <td>(128, 128, 3)</td>
                </tr>
                <tr>
                    <td>Conv2D</td>
                    <td>filters=128, kernel_size=(5, 5), activation="relu", 
                    kernel_regularizer=l2(0.0001)</td>
                </tr>
                <tr>
                    <td>MaxPooling</td>
                    <td>pool_size=(6, 6)</td>
                </tr>
                <tr>
                    <td>Conv2D</td>
                    <td>filters=128, kernel_size=(3, 3), activation="relu", 
                    kernel_regularizer=l2(0.0001)</td>
                </tr>
                <tr>
                    <td>MaxPooling</td>
                    <td>pool_size=(2, 2)</td>
                </tr>
                <tr>
                    <td>Conv2D</td>
                    <td>filters=128, kernel_size=(3, 3), padding="same", 
                    activation="relu", kernel_regularizer=l2(0.0001)</td>
                </tr>
                <tr>
                    <td>MaxPooling</td>
                    <td>pool_size=(2, 2)</td>
                </tr>
                <tr>
                    <td>Flatten</td>
                    <td>---</td>
                </tr>
                <tr>
                    <td>Dense</td>
                    <td>384, kernel_regularizer=l2(0.0001)</td>
                </tr>
                <tr>
                    <td>Dropout</td>
                    <td>0.3</td>
                </tr>
                <tr>
                    <td>Dense(output)</td>
                    <td>3, activation="softmax"</td>
                </tr>
            </table>

            [Compiling:]<br>
                [loss="categorical_crossentropy"]<br>
                [optimizer=keras.optimizers.Adam(learning_rate=0.001)]<br>
                [metrics=["accuracy"]]<br>
            """, unsafe_allow_html=True)

    st.image('assets/model_arch.PNG')

    eval, hyperparams = load_test_evaluation()
    st.write("Final model hyperparameters")
    model_df = pd.DataFrame.from_dict(hyperparams, orient='index')
    model_df.rename(columns={0: 'Value'}, inplace=True)
    st.dataframe(model_df)

    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/x_b_1/model_acc_x_b_1.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/x_b_1/model_losses_x_b_1.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(
        eval, index=['Loss', 'Accuracy'], columns=['Value']))

    st.write("### Detailed Performance on Test Set")

    # test_report = load_reports()['test']
    # st.code(f'''{test_report}''')
    col1, col2 = st.columns(2)
    test_heat_precision = plt.imread(
        f"outputs/x_b_1/pred_test_precision_heatmap_x_b_1.png")
    col1.image(test_heat_precision,
               caption='Prediction scores for Test set, recall')

    test_heat_classification = plt.imread(
        f"outputs/x_b_1/pred_test_classification_heatmap_x_b_1.png")
    col2.image(test_heat_classification,
               caption='Prediction scores for Test set, classification report')

    st.write("### Detailed Performance on Live Set")
    col1, col2 = st.columns(2)

    live_heat_precision = plt.imread(
        f"outputs/x_b_1/pred_live_precision_heatmap_x_b_1.png")
    col1.image(live_heat_precision,
               caption='Prediction scores for Live set, recall')

    live_heat_classification = plt.imread(
        f"outputs/x_b_1/pred_live_classification_heatmap_x_b_1.png")
    col2.image(live_heat_classification,
               caption='Prediction scores for Live set, classification report')

    st.success(
        """
        The model's scores meet the target values for all labels, which means
        that the developed model answers **Business Requirement 2**.
        """)

    st.markdown(
        """
        Please find detailed performance reports and test logs in the project's
        <a href=
             "https://github.com/RikaIljina/PP5/blob/main/README.md#final-model" 
             target="_blank" rel="noopener">README on GitHub</a>.
        """, unsafe_allow_html=True)

    st.write("#### Detailed Performance on Live Batches")

    st.write(
        """
        Business requirement 3 required an investigation into the possibility
        to reduce the risk of misclassifying a pet by letting the model make a
        summary prediction for a batch of images instead of trying to classify
        a pet based on one single image. It was assumed that a minimum
        threshold value is needed for the number of images to be batched before
        the first classification attempt should be made.\n\n
        A long series of trials was conducted where the model made predictions
        for batches of 2+n random live images for each label. The target F1
        score for batch classification was set at 1. The results are as
        follows:\n\n
        **Setup:**\n
        A script aggregated all live images of a label and yielded them in
        random combinations of 2+n images (a batch) to the classifier function.
        The classifier function calculated the mean probabilities for the batch
        and returned the dominant class without a confidence threshold to
        consider, meaning that any probability > 0.34 would result in a
        classification.\n
        The following boxplot graphs exemplify the final model’s results for 50
        batches at a time.\n
        Each batch consists of a given 2+n number of probability values for a
        specific class, as returned by the model. The box represents the upper
        and lower quartiles separated by the green median marker, while the red
        marker shows the mean value that we use for the classification.
        The circles represent outliers.\n
        _Please find below the graphs for batch sizes **5** and **15**_
        """)

    st.write("**Fin, 5 images/50 batches**")
    live_batch_fin = plt.imread(
        f"outputs/x_b_1/live_class_batch_probas_fin__x_b_1_50_5.png")
    st.image(live_batch_fin,
             caption='Label "fin", probability spread between batches of 5 images each')
    st.write("**Fin, 15 images/50 batches**")
    live_batch_fin = plt.imread(
        f"outputs/x_b_1/live_class_batch_probas_fin__x_b_1_50_15.png")
    st.image(live_batch_fin,
             caption='Label "fin", probability spread between batches of 15 images each')

    st.write("**Iris, 5 images/50 batches**")
    live_batch_iris = plt.imread(
        f"outputs/x_b_1/live_class_batch_probas_iris__x_b_1_50_5.png")
    st.image(live_batch_iris,
             caption='Label "iris", probability spread between batches of 5 images each')
    st.write("**Iris, 15 images/50 batches**")
    live_batch_iris = plt.imread(
        f"outputs/x_b_1/live_class_batch_probas_iris__x_b_1_50_15.png")
    st.image(live_batch_iris,
             caption='Label "iris", probability spread between batches of 15 images each')

    st.write("**Smilla, 5 images/50 batches**")
    live_batch_smilla = plt.imread(
        f"outputs/x_b_1/live_class_batch_probas_smilla__x_b_1_50_5.png")
    st.image(live_batch_smilla,
             caption='Label "smilla", probability spread between batches of 5 images each')
    st.write("**Smilla, 15 images/50 batches**")
    live_batch_smilla = plt.imread(
        f"outputs/x_b_1/live_class_batch_probas_smilla__x_b_1_50_15.png")
    st.image(live_batch_smilla,
             caption='Label "smilla", probability spread between batches of 15 images each')

    st.info(
        """
        As seen clearly on the figures, the red mean marker ascends with
        increasing batch size.\n\n
        For 5-image batches, the means are all above 0.7.\n
        For 15-image batches, the means are all above 0.8, with most being
        situated above 0.9.\n\n
        This trial was repeated multiple times with 500-1000 batches and 2-15
        images per batch.\n
        No confidence threshold was set.\n\n
        To summarize the findings:\n
        Batches of 4 or fewer images tend to misclassify at least one batch
        every 500 runs.\n
        Batches of 5 or more images have not shown any misclassification for an
        entire batch.\n\n
        """)
    st.success(
        """
        Running 500 batches per label with 5 images per batch resulted in an F1 
        score of 1 for each label, which answers **Business Requirement 3**.\n
        Please see the page "Recommendations" for adequate classification
        parameters.
        """)
