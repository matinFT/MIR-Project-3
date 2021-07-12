from selenium import webdriver
import time
import json
import numpy as np
import pandas as pd
import math


def get_article_info(driver):
    info = {"id": driver.current_url.split("/")[4]}
    title = driver.find_element_by_xpath('//h1[@class=\"name\"]').text
    info["title"] = title

    elements = driver.find_elements_by_xpath('//div[@class=\"count\"]')
    info["reference_count"] = elements[0].text
    info["citation_count"] = elements[1].text

    abstract = driver.find_element_by_css_selector("div.name-section > p").text
    info["abstract"] = abstract

    year = driver.find_element_by_css_selector("div.name-section > a > span.year").text
    info["date"] = int(year)

    authors = [x.text for x in driver.find_elements_by_css_selector("div.name-section > ma-author-string-collection > " +
                                                    "div.au-target > div.authors > div.author-item > a.author")]
    info["authors"] = authors
    click_button_element = driver.find_element_by_css_selector(
        "div.name-section > div.topics > ma-tag-cloud > div > div.show-more")
    driver.execute_script("arguments[0].click();", click_button_element)
    time.sleep(1)
    topics = driver.find_elements_by_css_selector("div.name-section > div.topics > ma-tag-cloud > div > ma-link-tag")
    related_topics = []
    for i in topics:
        related_topics.append(i.text)
    info["related_topics"] = related_topics

    info["references"] = []
    reference_ids = get_reference_ids(driver)
    for ref_id in reference_ids:
        info["references"].append(ref_id)
    return info


def get_reference_ids(driver):
    references = driver.find_elements_by_css_selector(
        "div.ma-paper-results > div.results > ma-card div.primary_paper > a.title")
    references_ids = []
    for ref in references:
        references_ids.append(ref.get_attribute("href").split("/")[4])
    return references_ids


def save_papers(number_of_papers):
    url_Queue = [
        "https://academic.microsoft.com/paper/2981549002",
        "https://academic.microsoft.com/paper/3105081694",
        "https://academic.microsoft.com/paper/2950893734",
        "https://academic.microsoft.com/paper/3119786062",
        "https://academic.microsoft.com/paper/2145339207",
        "https://academic.microsoft.com/paper/2153579005"
    ]
    crawled_ids = []
    driver = webdriver.Firefox()
    # driver = webdriver.Chrome(executable_path='E:\Downloads\Compressed\chromedriver.exe')
    pages_info = []
    i = 0
    while url_Queue and i < number_of_papers:
        current_page = url_Queue.pop(0)
        for j in range(2):
            try:
                driver.get(current_page)
                time.sleep(3)
                for x in range(2):
                    try:
                        page_info = get_article_info(driver)  # get article info has 1 second delay in it
                        pages_info.append(page_info)
                        ref_ids = page_info["references"]
                        current_page_id = current_page.split("/")[4]
                        for ref_id in ref_ids:
                            if ref_id not in crawled_ids and ref_id != current_page_id:
                                if "https://academic.microsoft.com/paper/{}".format(ref_id) not in url_Queue:
                                    url_Queue.append("https://academic.microsoft.com/paper/{}".format(ref_id))
                        i += 1
                        crawled_ids.append(current_page_id)
                        break

                    except:
                        if x == 1:
                            # print("error in get info")
                            raise Exception()
                        time.sleep(1)
                break

            except:
                if j == 1:
                    print("couldn't crawl ", current_page)
                pass

        if i % 5 == 0:
            print("{} papers crawled".format(i))

    with open('CrawledPapers.json', 'w') as outfile:
        json.dump(pages_info, outfile)


def cal_adj_matrix(papers_data, paper_to_index):
    adj_matrix = np.zeros((len(papers_data), len(papers_data)))
    for i in range(len(papers_data)):
        refs = papers_data[i]["references"]
        for ref in refs:
            j = paper_to_index.get(ref, -1)
            if j != -1:
                adj_matrix[i, j] = 1
    return adj_matrix


def save_page_rank(alpha, papers_file):
    papers_data = json.load(papers_file)
    paper_to_index = {}
    for index in range(len(papers_data)):
        paper_to_index[papers_data[index]["id"]] = index

    adj_matrix = cal_adj_matrix(papers_data, paper_to_index)
    P = adj_matrix.copy()
    V = np.zeros(len(P[0]))
    V.fill(1 / len(P[0]))
    V = np.matrix(V)
    #     return V
    for i in range(len(P)):
        if sum(P[i]) == 0:
            P[i] = V
        else:
            P[i] = P[i] / sum(P[i])
            P[i] = (1 - alpha) * P[i] + alpha * V
    #     return P
    P = np.matrix(P)
    res = V * P
    for i in range(1000):
        res = res * P

    with open("PageRank.json", "w") as page_rank_file:
        json.dump(dict([(i, res[0, paper_to_index[i]]) for i in paper_to_index]), page_rank_file)

    return page_rank_file


def cal_authors_ref_matrix(papers_data, authors_to_index):
    authors_ref_matrix = np.zeros((len(authors_to_index), len(authors_to_index)))
    for i in range(len(papers_data)):
        authors = papers_data[i]["authors"]
        refs = papers_data[i]["references"]
        for j in range(len(papers_data)):
            if i != j:
                if papers_data[j]["id"] in refs:
                    target_authors = papers_data[j]["authors"]
                    for a in authors:
                        for b in target_authors:
                            authors_ref_matrix[authors_to_index[a], authors_to_index[b]] = 1
    return authors_ref_matrix


def cal_hits_top_writers(n, papers_file):
    papers_data = json.load(papers_file)
    authors_to_index = {}
    for paper in papers_data:
        for author in paper["authors"]:
            if author not in authors_to_index:
                authors_to_index[author] = len(authors_to_index)

    authors_ref_matrix = cal_authors_ref_matrix(papers_data, authors_to_index)

    h = np.ones(len(authors_ref_matrix))
    a = np.ones(len(authors_ref_matrix))
    for rep in range(5):
        for i in range(len(authors_ref_matrix)):
            h[i] += a.reshape((1, len(a))).dot(authors_ref_matrix[i].reshape((len(authors_ref_matrix), 1)))

        for i in range(len(authors_ref_matrix)):
            a[i] += h.reshape((1, len(h))).dot(authors_ref_matrix[:, i])
        a = a / sum(a)
        h = h / sum(h)

    authorities = dict([(i, a[authors_to_index[i]]) for i in authors_to_index])
    return sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:n]


def recommend_paper(user_profile, papers_profiles):
    scores = {}
    valid_user_profile = sum(user_profile) != 0
    for paper_id in papers_profiles:
        if not valid_user_profile or sum(papers_profiles[paper_id]) == 0:
            scores[paper_id] = 0
            continue
        norm_factor = np.linalg.norm(user_profile, 2) * np.linalg.norm(papers_profiles[paper_id], 2)
        scores[paper_id] = user_profile.reshape((1, -1)).dot(
            papers_profiles[paper_id].reshape(-1, 1))[0, 0] / norm_factor

    # return sorted(scores.items(), key = lambda x: x[1], reverse = True)[:10]
    return [i[0] for i in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]


def cal_papers_profiles(papers_data, topics):
    papers_profiles = {}
    for paper in papers_data:
        vec = np.zeros(len(topics))
        for i in range(len(topics)):
            for topic in paper["related_topics"]:
                if topic.lower() == topics[i].lower():
                    vec[i] = 1
        papers_profiles[paper["id"]] = vec
    return papers_profiles


def corr(a, b):
    a_tilda = a - np.mean(a)
    b_tilda = b - np.mean(b)
    return a_tilda.dot(b_tilda) / (np.sqrt(sum(a_tilda ** 2)) * np.sqrt(sum(b_tilda ** 2)))


def pearson_similarity(a, b):  # a,b as np.array
    co_rated_items = (a != 0) * (b != 0)
    a_temp = a[co_rated_items]
    b_temp = b[co_rated_items]
    return corr(a_temp, b_temp)


# returns top n profiles with their similarity
# user_profiles, user_prof as np.array
def close_user_profiles(user_prof, user_profiles, n, predict_index):
    rated_fields = list(user_prof != 0)
    rated_fields[predict_index] = True
    scores = {}
    for i in range(len(user_profiles)):
        if 0 not in user_profiles[i, rated_fields]:
            scores[i] = pearson_similarity(user_prof, user_profiles[i])
    return [(user_profiles[x[0]], x[1]) for x in sorted(scores.items(),
                                                        key=lambda x: x[1], reverse=True)[:n]]


# puts user mean if no similar user found
def collaborative_filter(user_profile, n):
    user_profiles = pd.read_csv("data.csv")
    user_profiles.fillna(0, inplace=True)
    user_prof = np.array(user_profile).copy()
    user_mean = np.mean(user_profile)
    for i in range(len(user_prof)):
        if user_profile[i] == 0:
            top_similar_users = close_user_profiles(np.array(user_profile), np.array(user_profiles), n, i)
            user_prof[i] = user_mean
            s = 0
            if top_similar_users:
                for x in top_similar_users:
                    s += x[1] * (x[0][i] - np.mean(x[0]))
                user_prof[i] += s / sum([x[1] for x in top_similar_users])
    user_prof = user_prof / sum(user_prof)

    with open(papers_file_name, "r") as f:
        papers_profiles = cal_papers_profiles(json.load(f), list(user_profiles.columns))
        f.close()
    return user_prof, recommend_paper(user_prof, papers_profiles)


papers_file_name = "CrawledPapers.json"
while True:
    try:
        a = int(input("enter '1' for starting the crawler\nenter '2' to calculate and save pageRank\n" +
                      "enter '3' for calculating top writers\nenter '4' to recommend papers\n" +
                      "enter '5' to estimate user rates and recommend papers\n"))
    except:
        print("enter a number")
        continue

    if a == 1:
        save_papers(2000)

    elif a == 2:
        with open(papers_file_name, "r") as f:
            page_rank_file = save_page_rank(0.5, f)
            f.close()
            page_rank_file.close()
        print("pageRank saved in PageRank.json\n")

    elif a == 3:
        with open(papers_file_name, "r") as f:
            top_writers = cal_hits_top_writers(10, f)
            f.close()
        print(pd.DataFrame({"writer": [x[0] for x in top_writers],
                            "score": [x[1] for x in top_writers]}))
        print("\n")

    elif a == 4:
        user_profiles = pd.read_csv("data.csv")
        user_profiles.fillna(0, inplace=True)
        topics = list(user_profiles.columns)
        while True:
            try:
                user_index = int(input("enter user's index whom you want to recommend\n"))
                break
            except:
                continue
        with open(papers_file_name, "r") as f:
            papers_profiles = cal_papers_profiles(json.load(f), topics)
            f.close()
        print(recommend_paper(np.array(user_profiles.loc[user_index, :]), papers_profiles))
        print("\n")

    elif a == 5:
        while True:
            try:
                user_index = int(input("enter user's index whom you want to collaborative_filter\n"))
                break
            except:
                continue
        user_profiles = pd.read_csv("data.csv")
        user_profiles.fillna(0, inplace=True)
        pred_profile, recommended = collaborative_filter(user_profiles.iloc[user_index, :], 10)
        print(pd.DataFrame({
            "before": list(user_profiles.iloc[user_index, :]),
            "after": list(pred_profile)
        }))
        print(recommended)
        print("\n")
    else:
        print("enter an integer between 1 and 5")
