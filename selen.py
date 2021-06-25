from selenium import webdriver
import time
import json
import numpy as np


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

    authors = driver.find_element_by_css_selector("div.name-section > ma-author-string-collection > " +
                                                  "div.au-target > div.authors").text.split(" , ")
    info["authors"] = authors

    click_button_element = driver.find_element_by_css_selector("div.name-section > div.topics > ma-tag-cloud > div > div.show-more")
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
    pages_info = []
    i = 0
    while url_Queue and i < number_of_papers:
        current_page = url_Queue.pop(0)
        driver.get(current_page)
        time.sleep(2)
        # time.sleep(4)
        page_info = get_article_info(driver)
        pages_info.append(page_info)
        ref_ids = page_info["references"]
        for ref_id in ref_ids:
            if ref_id not in crawled_ids:
                if "https://academic.microsoft.com/paper/{}".format(ref_id) not in url_Queue:
                    url_Queue.append("https://academic.microsoft.com/paper/{}".format(ref_id))
        i += 1

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
    p = adj_matrix.copy()
    v = np.zeros(len(p[0]))
    v.fill(1 / len(p[0]))
    for i in range(len(p)):
        if sum(p[i]) == 0:
            p[i] = v
        else:
            p[i] = (1 - alpha) * p[i] + alpha * v

    res = v.reshape((1, len(v))).dot(p)
    for i in range(10):
        res = res.dot(p)

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


# save_papers(10)


# with open("content.json", "r") as f:
#     page_rank_file = save_page_rank(0.5, f)


with open("content.json", "r") as f:
    top_writers = cal_hits_top_writers(10, f)
    print(top_writers)
    # problem with writer with '' name!
