from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def main():
    stop_words = set(STOPWORDS)
    stop_words.update(ENGLISH_STOP_WORDS)
    #extra stop words
    extra_words=["said","say","seen","come","end","came","year","years","new","saying"]
    stop_words = ENGLISH_STOP_WORDS.union(extra_words)

    df = pd.read_csv('/kaggle/input/question1/train.csv')
#     df=df.head(n=1000)



    cat_business = []
    cat_entertainment = []
    cat_health = []
    cat_technology = []
    #store the content for each category
    for index in range(len(df.Label)):
        cat = df.Label[index]
        if cat == "Business":
            cat_business.append(df.Content[index])
        elif cat == "Entertainment":
            cat_entertainment.append(df.Content[index])
        elif cat == "Health":
            cat_health.append(df.Content[index])
        elif cat == "Technology":
            cat_technology.append(df.Content[index])

    str_bus = ''.join(cat_business)
    str_ent = ''.join(cat_entertainment)
    str_hea = ''.join(cat_health)
    str_tec = ''.join(cat_technology)

    #produce wordcloud for each category
    cloud = WordCloud(stopwords=stop_words)

    w = cloud.generate(str_bus)
    plt.figure()
    plt.imshow(w)
    plt.title("Business")
    plt.axis("off")
    plt.savefig('/kaggle/working/Business.png')

    w = cloud.generate(str_ent)
    plt.figure()
    plt.imshow(w)
    plt.title("Entertainment")
    plt.axis("off")
    plt.savefig('/kaggle/working/Entertainment.png')

    w = cloud.generate(str_hea)
    plt.figure()
    plt.title("Health")
    plt.imshow(w)
    plt.axis("off")
    plt.savefig('/kaggle/working/Health.png')

    w = cloud.generate(str_tec)
    plt.figure()
    plt.imshow(w)
    plt.title("Technology")
    plt.axis("off")
    plt.savefig('/kaggle/working/Technology.png')
	
if __name__ == "__main__":
    main()