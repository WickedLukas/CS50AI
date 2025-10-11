import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    transition_model(corpus, list(sorted(corpus.keys()))[1], DAMPING)

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    N = len(corpus)
    p_rand = (1 - damping_factor) / N

    links = corpus[page]
    links_num = len(links)

    dist = {}
    if links_num == 0:
        for page in corpus:
            dist[page] = 1 / N
        return dist

    for page in corpus:
        dist[page] = p_rand
        if page in links:
            dist[page] += damping_factor / links_num

    return dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {}
    for page in corpus:
        page_rank[page] = 0

    page = random.choice(list(corpus.keys()))
    page_rank[page] = 1

    if n == 1:
        return page_rank

    for _ in range(n - 1):
        dist = transition_model(corpus, page, damping_factor)
        pages, probabilities = zip(*dist.items())
        page = random.choices(pages, probabilities)[0]
        page_rank[page] += 1

    for page in page_rank:
        page_rank[page] /= n

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    page_rank = {}
    for page in corpus:
        page_rank[page] = 1 / N

    run = True
    p_rand = (1 - damping_factor) / N
    page_rank_last = copy.copy(page_rank)
    while(run):
        for page in corpus:
            sum_linked = 0
            for other_page in corpus:
                if len(corpus[other_page]) == 0:
                    sum_linked += page_rank_last[other_page] / N
                elif(page in corpus[other_page]):
                    sum_linked += page_rank_last[other_page] / len(corpus[other_page])

            page_rank[page] = p_rand + damping_factor * sum_linked

        run = False
        for page in corpus:
            if abs(page_rank[page] - page_rank_last[page]) > 0.001:
                run = True

        page_rank_last = copy.copy(page_rank)

    return page_rank


if __name__ == "__main__":
    main()
