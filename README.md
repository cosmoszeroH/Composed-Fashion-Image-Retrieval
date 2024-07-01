# CLIP-Combiner-Composed-Image-Retrieval

In this repo, I post a simple demo to retrieve fashion image from a reference query and a text that includes a descriptive request from the user about the image.

The project folder must be like this:
project_base_path
└─── Dataset
     └───  fashionIQ_dataset
          └─── captions
                | cap.dress.test.json
                ...
                | cap.toptee.val.json
          └─── image_splits
                | split.dress.test.json
                ...
                | split.toptee.val.json
          └─── images
                | 245600258X.jpg
                ...
└─── demo_feature
└─── src
| demo.py
| extract_features.py

First, you have to have FashionIQ dataset in fashionIQ_dataset and have all the path like above

Second, you run extract_features.py file to extract feature in advance

Finally, **streamlit run demo.py** in your terminal to run the demo
