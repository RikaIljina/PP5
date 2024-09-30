import streamlit as st
import os
import matplotlib.pyplot as plt
import itertools
from src.data_visualization.get_montage import image_montage
import joblib


def page_dataset_assessment_body():
    output_path = os.path.relpath('outputs')
    full_dataset_path = os.path.relpath('inputs/datasets/pets_slim')
    LABELS = sorted(joblib.load(f"{output_path}/class_dict.pkl").values())
    COMBOS = list(itertools.combinations(LABELS, 2))
    
    
    st.write("### Dataset Assessment")
    st.info(
        '''The client is interested in a recommendation based on the comparison of the 
            image labels in the dataset to be able to evaluate and adjust future training
            sets to work with our model.''')
    
    st.write("#### Average and variance images for each label")
    if st.checkbox("Show", key="show-1"):
      
        avg_grid = plt.imread(os.path.join(output_path, 'average_images_grid.png'))
        st.image(avg_grid, caption="The average images for each label in the 'train' subset")
        
        var_grid = plt.imread(os.path.join(output_path, 'variance_images_grid.png'))
        st.image(var_grid, caption="The variance images for each label in the 'train' subset")
        
        var_grid_norm = plt.imread(os.path.join(output_path, 'variance_normalized_images_grid.png'))
        st.image(var_grid_norm, caption="The normalized variance images for each label in the 'train' subset")

        st.success(
        f"* We notice that the average image for each pet is clearly distinguishable "
        f"from the others.\n\n"
        f"* We also notice that the 'iris' label might contain too many similar images, "
        f"seen as clear outlines on the average image. This might lead to bias during "
        f"model training.\n\n"
        f"* The variance images show the mean variance between each image in the label "
        f"set, brighter areas indicating greater variance. Dark areas can be interpreted "
        f"as unchanging aspects such as floor or wall."
        )

        st.write("---")

    st.write("#### Differences between averages for each label combo")
    if st.checkbox("Show", key="show-2"):
        for combo in COMBOS:
            diff_between_avgs = plt.imread(f"{output_path}/average_imgs_{'_'.join(combo)}.png")
            st.image(diff_between_avgs)

        st.success(
        f"* The differences between the average images show by how much each pixel in "
        f"one mean image differs from the same pixel in the other mean image. "
        f"Bright areas indicate greater differences. \n\n"
        f"* We notice that there are visible differences between all labels, with the "
        f"**'fin - smilla'** comparison showing the largest bright area. \n\n"
        f"* We make the preliminary conclusion that the labels **'fin'** and **'smilla'**"
        f" might turn out to be easiest for the model to distinguish."
        )

    st.write("#### Histogram comparison")
    if st.checkbox("Show", key="show-3"):
        st.info(
        f"On top of the visual comparison, we want to compare the histograms for the "
        f"average images of each combo and use the results to evaluate our dataset.\n"
        f"Seeing as we will be training our model on 3-channel RGB color images, we will "
        f"compare each channel separately."
        )
        st.write(f"---")
        st.write('##### Create baseline')
        st.info(
        f"To create the baselines, we have loaded at least 200 random non-consecutive "
        f"images from each label, split the resulting array in half and compared the "
        f"means of both halves with each other by subtracting one from the other and "
        f"converting the result to absolute values.\n"
        f"- Dark pixels: there is little to no difference in hue and/or brightness "
        f"between two compared pixels\n"
        f"- Bright pixels: there is noticeable difference in hue and/or brightness "
        f"between two compared pixels\n\n"
        )
        for label in LABELS:
            baseline = plt.imread(f"{output_path}/baseline_imgs_{label}.png")
            st.image(baseline)
        st.info(
        f"The resulting dark, noisy images represent the amount of variance we can expect"
        f" from two indistinguishable animals.\n"
        )
        st.write("---")
        
        st.info(f"Let's assess the histograms where we compare each label to itself:")
        for label in LABELS:
            baseline_diffs = plt.imread(f"{output_path}/hist_baseline_average_{label}_rgb.png")
            st.image(baseline_diffs)
            
        st.write('##### Compare all labels')
        st.info(f"Let's now compare the histograms for each label combination:")
        
        for combo in COMBOS:
            hist_diffs = plt.imread(f"{output_path}/hist_average_{'_'.join(combo)}_rgb.png")
            st.image(hist_diffs)

        st.success(
        f"We can clearly see that the histograms depicting the differences between two "
        f"different labels are much more pronounced than the baseline histogram diffs. "
        f"This reinforces our initial assumption that the pets seem to have prominent "
        f"features that make them distinguishable from one another.\n\n"
        f"However, we want to go one step further and find metrics that will inform our "
        f"training set evaluation even better."
        )

    st.write('#### Metrics analysis')
    if st.checkbox("Show", key="show-4"):
        
        st.write('##### Analyze the comparison metrics')

        st.info(
        f"In order to assess the similarity between our image sets, we performed "
        f"histogram comparison for each color channel using five distinct methods and "
        f"summarized the results in the following heatmap."
        )
        
        st.warning(
        f"##### NB!\n\n"
        f"The annotated values on the heatmap are normalized within the range (0, 1) to "
        f"denote the similarity between the datasets and do not represent the actual "
        f"calculated values of the applied metric."
        )
        
        heat_ch = plt.imread(f"{output_path}/heatmap_by_channel.png")
        st.image(heat_ch)

        st.success(
            f"As expected, the upper half of the heatmap containing our baseline data "
            f"is showing overwhelmingly high similarity values.\n\n"
            f"At the same time, . "
            f""
            )
        col1, col2 =st.columns([5, 4])
        heat = plt.imread(f"{output_path}/heatmap_conclusion.png")
        col1.image(heat)

        heat_mm = plt.imread(f"{output_path}/heatmap_mean_med.png")
        col2.image(heat_mm)
        
        with st.expander('Metrics details'):
            st.markdown(
            f"""<b>Methods:</b>
            <ul>
            <li>
            <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient" target="_blank">
            <b>Correlation</b></a>:
            <br>Computes the correlation coefficient between two histograms, measuring
            the strength of a linear relationship between two histograms. Values range
            from -1 (perfectly anti-correlated) to 1 (perfectly correlated).<br>
            A value of 0 means no correlation."
            </li>
            <li>
            <a href="https://en.wikipedia.org/wiki/Chi-squared_test" target="_blank">
            <b>Chi-Squared</b></a>:
            This method measures the similarity between two histograms by calculating 
            the sum of the squared differences normalized by the values of the histograms.
            It is sensitive to small changes in the histogram bins.<br>
            The result is a value between 0 and infinity, where 0 means highest similarity.
            </li>
            <li>
            <a href="https://blog.datadive.net/histogram-intersection-for-change-detection/" target="_blank">
            <b>Intersection</b></a>:
            Calculates the sum of the minimum values of corresponding bins in two histograms.
            The result is a value between 0 and 1, where 1 means the histograms are identical.
            </li>
            <li>
            <a href="https://en.wikipedia.org/wiki/Bhattacharyya_distance" target="_blank">
            <b>Bhattacharyya</b></a>:
            The Bhattacharyya distance quantifies the overlap between two probability
            distributions. It is useful for comparing two probability histograms and
            provides a measurement of the distance between two distributions.
            The result is a value between 0 and infinity, where 0 means highest similarity.
            </li>
            <li>
            <a href="https://en.wikipedia.org/wiki/Bhattacharyya_distance" target="_blank">
            <b>Euclidean Distance</b></a>:
            This method measures the straight-line distance between corresponding bins in
            two histograms. It sums the squared differences of each bin and takes the
            square root. The smaller the distance, the more similar the histograms are.
            The result is a value between 0 and infinity, where 0 means highest similarity.
            </li>
            </ul>"""
            , unsafe_allow_html=True
            )

    st.write("---")

    st.write("#### Image Montage")
    if st.checkbox("Show", key="show-5"):
        st.info("To refresh the montage, click on the 'Create Montage' button")
        labels = os.listdir(f'{full_dataset_path}/train')
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)
        show_all = st.checkbox("Show all")
        if st.button("Create Montage"):
            image_montage(f'{full_dataset_path}/train', label_to_display,  
                            3, 3, show_all, figsize=(10,10))
        
        st.write("---")
        
                
