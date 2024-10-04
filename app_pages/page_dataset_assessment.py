import streamlit as st
import os
import matplotlib.pyplot as plt
import itertools
from src.data_visualization.get_montage import image_montage
import joblib
import pandas as pd
import numpy as np



def page_dataset_assessment_body():
    output_path = os.path.relpath("outputs")
    full_dataset_path = os.path.relpath("inputs/datasets/pets_slim")
    LABELS = sorted(joblib.load(f"{output_path}/class_dict.pkl").values())
    COMBOS = list(itertools.combinations(LABELS, 2))

    st.write("### Dataset Assessment")
    st.markdown(
        """
        <div class="blue-div">
          <h5>Business requirement 1:</h5>
            <p>The client is interested in a recommendation regarding the scale
            and quality of future datasets as well as an investigation of a 
            correlation between the similarity of the pets' visual features and
            the performance of the model.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )

    st.write("\n\n")

    st.markdown(
        """
        <h5 style="text-decoration: underline;">
        Initial visual assessment
        </h5>
        """,
        unsafe_allow_html=True,
    )
    if st.checkbox("Show", key="show-0"):
        
        col1, col2, col3 = st.columns(3)
        col1.image('inputs/datasets/pets_slim/train/fin/train_fin_0043.png', 
                   caption='Fin')
        col2.image('inputs/datasets/pets_slim/train/iris/train_iris_0076.png',
                   caption='Iris')
        col3.image('inputs/datasets/pets_slim/train/smilla/train_smilla_0261.png',
                   caption='Smilla')
        
        st.info(
            """
            * By looking through the images in the dataset, we notice that the
            pet 'smilla', being rather small, usually takes up too little space
            on the images, with the background being dominant.
            * The 'fin' label shows sufficient presence and variety.
            * Since the fur of the pet 'iris' has light and dark patches while 
            the pet 'smilla' is predominantly black and the pet 'fin' is 
            completely beige-white, we assume that Iris's unique features 
            might overlap with the other pets.
            """)
    
    st.markdown(
        """
        <h5 style="text-decoration: underline;">
        Average and variance images for each label
        </h5>
        """,
        unsafe_allow_html=True,
    )

    if st.checkbox("Show", key="show-1"):

        avg_grid = plt.imread(
            os.path.join(output_path, "average_images_grid.png")
        )
        st.image(
            avg_grid,
            caption="The average images for each label in the 'train' subset",
        )

        var_grid = plt.imread(
            os.path.join(output_path, "variance_images_grid.png")
        )
        st.image(
            var_grid,
            caption="The variance images for each label in the 'train' subset",
        )

        var_grid_norm = plt.imread(
            os.path.join(output_path, "variance_normalized_images_grid.png")
        )
        st.image(
            var_grid_norm,
            caption="The normalized variance images for each label in the \
                'train' subset",
        )

        st.success(
            """
            The variance images show the mean variance between each image in 
            the label set, brighter areas indicating greater variance.
            Dark areas can be interpreted as unchanging aspects such as floor 
            or wall.
            
            * We notice that the average image for each pet is clearly
            distinguishable from the others.\n\n
            * We also notice that the 'iris' label might contain too many
            similar images, seen as clear outlines on the average image. This 
            might lead to bias during model training.\n\n
            """
        )

        st.write("---")

    st.markdown(
        """
                <h5 style="text-decoration: underline;">
                Differences between averages for each label combo
                </h5>
                """,
        unsafe_allow_html=True,
    )

    if st.checkbox("Show", key="show-2"):
        for combo in COMBOS:
            diff_between_avgs = plt.imread(
                f"{output_path}/average_imgs_{'_'.join(combo)}.png"
            )
            st.image(diff_between_avgs)

        st.success(
            """
            The differences between the average images show by how much each 
            pixel in one mean image differs from the same pixel in the other 
            mean image. Bright areas indicate greater differences. \n\n
            * We notice that there are visible differences between all labels, 
            with the **'fin - smilla'** comparison showing the largest bright 
            area. \n\n
            * We make the preliminary conclusion that the labels **'fin'** and 
            **'smilla'** might turn out to be easiest for the model to 
            distinguish.
            * The difference images for the other two pets can not be
            satisfactorily assessed in that manner.
            * We calculate the Mean and Variance values for the difference 
            images, receiving the following results:\n
            """
        )
        
        dict1 = {"Mean for ('fin', 'iris')": 0.08560952,
            "Mean for ('fin', 'smilla')":     0.14524248,
            "Mean for ('iris', 'smilla')":    0.0888353,
            "Variance for ('fin', 'iris')":   0.0073880,
            "Variance for ('fin', 'smilla')": 0.0107992,
            "Variance for ('iris', 'smilla')": 0.0040149}
        st.dataframe(pd.DataFrame(dict1, index=(['Value'])).T)
        
        st.success(
            """
            While the Mean values seem inconclusive for the “fin – iris” and
            “iris – smilla” pairs, the Variance values representing the
            variance for each pixel in the Mean images seem to show a clear
            trend, with “iris – smilla” showing the least variance, followed
            by “fin – iris” and finally with “fin - smilla”
            at the top.\n\n
            We make a preliminary conclusion that the images in the sets "fin"
            and "smilla" will have the least error rates, followed by "fin" and
            "iris" and finally with "smilla" and "iris" having the most 
            misclassifications between them.
            """)

    st.markdown(
        """
            <h5 style="text-decoration: underline;">
            Histogram comparison
            </h5>
            """,
        unsafe_allow_html=True,
    )
    if st.checkbox("Show", key="show-3"):
        st.info(
            """
            On top of the visual comparison, we want to compare the histograms 
            of the average images of each combo and use the results to
            evaluate our dataset.\n
            Seeing as we will be training our model on 3-channel RGB color
            images, we will compare each channel separately.
            """
        )
        st.write(f"---")
        st.write("##### Create baseline")
        st.info(
            """
            To create the baselines, we have loaded at least 200 random 
            non-consecutive images from each label, split the resulting array
            in half and compared the means of both halves with each other by
            subtracting one from the other and converting the result to
            absolute values.\n
            - Dark pixels: there is little to no difference in hue and/or
            brightness between two compared pixels\n
            - Bright pixels: there is noticeable difference in hue and/or
            brightness between two compared pixels
            """
        )
        for label in LABELS:
            baseline = plt.imread(f"{output_path}/baseline_imgs_{label}.png")
            st.image(baseline)
        st.info(
            """
            The resulting dark, noisy images represent the amount of variance
            we can expect from two indistinguishable animals.
            """
        )
        st.write("---")

        st.info(
        f"Let's assess the histograms where we compare each label to itself:"
        )
        for label in LABELS:
            baseline_diffs = plt.imread(
                f"{output_path}/hist_baseline_average_{label}_rgb.png"
            )
            st.image(baseline_diffs)

        st.write("##### Compare all labels")
        st.info(
            f"Let's now compare the histograms for each label combination:"
        )

        for combo in COMBOS:
            hist_diffs = plt.imread(
                f"{output_path}/hist_average_{'_'.join(combo)}_rgb.png"
            )
            st.image(hist_diffs)

        st.success(
            """
            We can clearly see that the histograms depicting the differences 
            between two different labels are much more pronounced than the
            baseline histogram diffs.\n
            This reinforces our initial assumption that the pets seem to have
            prominent features that make them distinguishable from one another.
            \n\n
            However, we want to go one step further and find metrics that will
            inform our training set evaluation even better.
            """
        )

    st.markdown(
        """
        <h5 style="text-decoration: underline;">
        Metrics analysis
        </h5>
        """,
        unsafe_allow_html=True,
    )
    if st.checkbox("Show", key="show-4"):

        st.write("##### Analyze the comparison metrics")

        st.info(
            """
            In order to assess the similarity between our image sets, we 
            performed histogram comparison for each color channel using five
            distinct methods and summarized the results in the following
            heatmap."
            """
        )

        st.warning(
            """
            ##### NB!\n\n
            The annotated values on the heatmap are normalized within the range
            (0, 1) to denote the similarity between the datasets and do not
            represent the actual calculated values of the applied metric.
            """
        )

        heat_ch = plt.imread(f"{output_path}/heatmap_by_channel.png")
        st.image(heat_ch)

        st.success(
            """
            As expected, the upper half of the heatmaps containing our baseline
            metrics is showing overwhelmingly high similarity values, while the
            lower half with the pet comparisons is showing low similarity 
            values.\n\n
            At the same time, there seem to be a stark variance between 
            different combinations, reinforcing our assumption that certain
            pet images have more overlap in color and shape than others.\n\n
            """
        )
        col1, col2 = st.columns([5, 4])
        heat = plt.imread(f"{output_path}/heatmap_conclusion.png")
        col1.image(heat)

        heat_mm = plt.imread(f"{output_path}/heatmap_mean_med.png")
        col2.image(heat_mm)

        st.success(
            """
            A first glance at the heatmap and the mean values per metric as 
            well as the overall mean and median values paint a clear picture:
            \n\n
            - The **Fin**-**Smilla** pair seems to be very distinguishable, 
            while **Iris** has more overlap with the other two image sets.\n\n
            - Surprisingly, the **Intersection** metric for the **Fin**
            baseline values shows less similarity than for the other baseline
            values. This might point to the dataset not being sufficiently 
            balanced in itself, with too much variation in poses and lighting 
            between the images. This finding warrants further investigation.
            \n\n
            - The overlap between the cats **Smilla** and **Iris** might be
            amplified by the fact that they are roughly the same size and shape
            and take up only a small fraction of the overall image. Thus, the
            wall and floor pixels are dominant in every image.
            """
        )
        st.info(
            """
            So far, the methods suggest the following order of the pet pairs on
            the similarity scale:
            """)
        st.dataframe(pd.DataFrame(columns=['Different', '--------', 'Similar'],
                                  index=['Correlation', 'Chi-Squared',
                                         'Intersection', 'Bhattacharyya',
                                         'Euclidean Distance'],
                                  data=[['Fin - Smilla', 'Iris - Smilla',
                                        'Fin - Iris'], 
                                        ['Fin - Smilla', 'Iris - Smilla',
                                        'Fin - Iris'],
                                        ['Fin - Smilla', 'Fin - Iris',
                                         'Iris - Smilla'], 
                                        ['Fin - Smilla', 'Fin - Iris',
                                         'Iris - Smilla'],
                                        ['Iris - Smilla', 'Fin - Smilla',
                                        'Fin - Iris'], ]))

        with st.expander("Comparison methods in detail"):
            st.markdown(
                f"""<b>Methods:</b>
            <ul>
            <li>
            <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient" target="_blank">
            <b>Correlation</b></a>:
            <br>Computes the correlation coefficient between two histograms,
            measuring the strength of a linear relationship between two
            histograms. Values range from -1 (perfectly anti-correlated) to 1
            (perfectly correlated).<br> A value of 0 means no correlation."
            </li>
            <li>
            <a href="https://en.wikipedia.org/wiki/Chi-squared_test" target="_blank">
            <b>Chi-Squared</b></a>:
            <br>This method measures the similarity between two histograms by calculating 
            the sum of the squared differences normalized by the values of the histograms.
            It is sensitive to small changes in the histogram bins.<br>
            The result is a value between 0 and infinity, where 0 means highest similarity.
            </li>
            <li>
            <a href="https://blog.datadive.net/histogram-intersection-for-change-detection/" target="_blank">
            <b>Intersection</b></a>:
            <br>Calculates the sum of the minimum values of corresponding bins in two histograms.
            The result is a value between 0 and 1, where 1 means the histograms are identical.
            </li>
            <li>
            <a href="https://en.wikipedia.org/wiki/Bhattacharyya_distance" target="_blank">
            <b>Bhattacharyya</b></a>:
            <br>The Bhattacharyya distance quantifies the overlap between two probability
            distributions. It is useful for comparing two probability histograms and
            provides a measurement of the distance between two distributions.
            The result is a value between 0 and infinity, where 0 means highest similarity.
            </li>
            <li>
            <a href="https://en.wikipedia.org/wiki/Bhattacharyya_distance" target="_blank">
            <b>Euclidean Distance</b></a>:
            <br>This method measures the straight-line distance between corresponding bins in
            two histograms. It sums the squared differences of each bin and takes the
            square root. The smaller the distance, the more similar the histograms are.
            The result is a value between 0 and infinity, where 0 means highest similarity.
            </li>
            </ul>""",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
            <h5 style="text-decoration: underline;">
            Conclusions and recommendations for dataset compilation
            </h5>
            """,
        unsafe_allow_html=True,
    )
    if st.checkbox("Show", key="show-5"):
        st.info(
            """
            In order to avoid high bias, low variance between the different labels and
            too high variance within a single label, the following recommendations for
            compiling the training dataset should be followed:\n\n
            * The more images a label contains, the better.\n
            * Images should be taken at different times of the day with different lighting
            to ensure that every possible lighting condition that the device will encounter
            during live classification is present in the train set.\n
            * At the same time, over- and underexposure should be avoided to 
            make sure that certain features are not rendered unrecognizable.\n
            * If possible, the images should show more of the pet than of the background.\n
            * If the pets use to wear different collars or clothes, the images should
            reflect that in appropriate proportions.\n
            * The pets should behave naturally on the images, just as they will be during
            the capturing of live images.\n
            * The distance between the camera and the pet should reflect the actual placement
            of camera and feeding device in the real-world application of the device.\n
            * The model should be trained with images from the same camera type and the
            same size/resolution that will be used during live classification.\n\n
            When analyzing image data, the following aspects should be considered:\n\n
            * The mean and variation images of a dataset should not contain clear
            outlines of the pet; this is a sign of an excess number of highly
            similar images, for example of the pet sitting there without moving.\n
            * Broadly speaking, regarding the metrics, we want the similarity values in the
            baseline part of the heatmap to go up and be close to 1, while we want to
            see the values in the label comparison part reduced to as close to 0 as
            possible. However, a pronounced difference between the two parts is already
            a good sign that the model will pick up on the differences, given a sufficient
            amount of data to train on.
            """
        )

    st.write("---")

    st.markdown(
        """
            <h5 style="text-decoration: underline;">
            Image Montage
            </h5>
            """,
        unsafe_allow_html=True,
    )
    if st.checkbox("Show", key="show-6"):
        st.info("To refresh the montage, click on the 'Create Montage' button")
        labels = os.listdir(f"{full_dataset_path}/train")
        label_to_display = st.selectbox(
            label="Select label", options=labels, index=0
        )
        show_all = st.checkbox("Show all")
        if st.button("Create Montage"):
            image_montage(
                f"{full_dataset_path}/train",
                label_to_display,
                3,
                3,
                show_all,
                figsize=(10, 10),
            )

        st.write("---")
