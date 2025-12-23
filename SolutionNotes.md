# Solution Notes

This was assignement in Kiu.

## Repositories - Neetcode Problems, Sort Algorithms

First repo came to my mind - leetcode problems.
Which subset of leetcode problems? Obviously, the ones from - Neetcode.io.

Second repo was sort problems. 

Pretty easy to understand problmes. Plus, there is an actual real problem
of detecting leetcode plagirism. 

I wrote the code to clone them.
I was going to clone 5 repos initially, thats why this code handles a list of repos. 

```python
import httpx
import asyncio
from pathlib import Path


async def fetch_github_repo_content(url: str) -> dict:
    headers = {
        "Accept": "application/vnd.github.object",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    return {}

async def fetch_file_content_from_download_url(download_url: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        response = await client.get(download_url)
        response.raise_for_status()
        return response.text
    return ""

async def fetch_list_of_file_name_content_tuples(github_repo_content: dict, max_concurrent: int = 3):
    python_files = [item for item in github_repo_content['entries'] if item['name'].endswith('.py')]
    semaphore = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient() as client:
        content_tasks = []
        for pf in python_files:
            content_task = fetch_file_content_from_download_url(
                pf['download_url'],
                client,
                semaphore
            )
            content_tasks.append(content_task)

        contents = await asyncio.gather(*content_tasks)

    name_content_tuples = list(zip([pf['name'] for pf in python_files], contents))
    return name_content_tuples


def write_python_files(relative_path: str, name_content_tuples: list) -> None:
    output_dir = Path.cwd() / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in name_content_tuples:
        file_path = output_dir / name
        file_path.write_text(content, encoding='utf-8')


async def fetch_urls_and_save_in_dirs():
    urls = [
        "https://api.github.com/repos/neetcode-gh/leetcode/contents/python?ref=main",
        "https://api.github.com/repos/TheAlgorithms/Python/contents/sorts?ref=master"
    ]

    dir_names = ['neetcode', 'sorts']
    for i in range(len(dir_names)):
        url = urls[i]
        dir_name = dir_names[i]

        repo = await fetch_github_repo_content(url)
        names_contents = await fetch_list_of_file_name_content_tuples(repo)
        write_python_files(dir_name, names_contents)

await fetch_urls_and_save_in_dirs()
```

## Chunking Implementation

Then I implemented chunking using the library treesitter. 

``` python 
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from pathlib import Path
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def chunk_python_code_in_functions_and_classes(code:str):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    function_node = root_node.children
    functions_and_classes = [f for f in function_node if f.type in ['function_definition', 'class_definition']]
    return [fc.text for fc in functions_and_classes]


def get_chunks_of_fuctions_and_classes_from_dir(dir_path):
    base = Path(dir_path)
    result = []
    for file in base.rglob("*"):
        if file.is_file() and file.suffix == '.py':
            code = file.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_python_code_in_functions_and_classes(code)
            result.extend(chunks)
    return result
```


## Embedding 

Then I tried to embedd fucions

First I tried 7xx length vector embedding model "Alibaba" something. It was trash.
Then I tried gemini-emebedding-001:

```python 
from google import genai
from google.colab import userdata
from google.genai import types
google_key = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=google_key)


def generate_embeddings(chunks:list[str]):
  result = client.models.embed_content(
          model="gemini-embedding-001",
          contents=chunks,
          config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
          )
  embeddings = np.array([embedding.values for embedding in result.embeddings])
  embeddings = embeddings.astype('float32')

  return embeddings


chunks = get_chunks_of_fuctions_and_classes_from_dir('sorts')
embeddings = generate_embeddings(chunks)


import numpy as np, faiss
dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)
metric = faiss.METRIC_INNER_PRODUCT
M=32
base = faiss.IndexHNSWFlat(dim, M, metric)
base.hnsw.efConstruction = 200
base.hnsw.efSearch = 64
index = faiss.IndexIDMap2(base)
ids = [i for i in range(embeddings.shape[0])]
ids = np.array(ids)
index.add_with_ids(embeddings, ids)
```

And test the embeddings:

```python
code = """

def bucket_sort(my_list: list, bucket_count: int = 10) -> list:
    ""
    >>> data = [-1, 2, -5, 0]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> data = [9, 8, 7, 6, -12]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> data = [.4, 1.2, .1, .2, -.9]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> bucket_sort([]) == sorted([])
    True
   2, 2, 1, 1, 3]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> data = [5, 5, 5, 5, 5]
"""
search_chunks = [code]
search_embed = generate_embeddings(search_chunks)
faiss.normalize_L2(search_embed)
D, I = index.search(search_embed, k=5)

print(D,I)
```

It is works amazing. 



Then I decided I was not going to spend any more time on this exercise, and focused all my time and energy with my  chatbot projects at Widgera. 

About a month later, I decided that I am not giving up anything I start on the half way. 
I am not starting any personal project till I finish this one. 

Lets go. 

## 2 positive and 2 negative examples:

### Stalin Sort

Obviously, I was gonna pick this one. 

**Positive (Plagiarized):**
```python
def stalin_sort(sequence: list[int]) -> list[int]:
    output = [sequence[0]]
    for val in sequence[1:]:
        if val >= output[-1]:
            output.append(val)
    return output
```

**Negative (Original):**
```python
def stalin_sort(sequence: list[int]) -> list[int]:
    if not sequence:
        return []
    
    filtered = [sequence[0]]
    prev = sequence[0]
    
    for i in range(1, len(sequence)):
        if sequence[i] >= prev:
            filtered.append(sequence[i])
            prev = sequence[i]
    
    return filtered
```

### Quick Sort

**Positive (Plagiarized):**
```python
from random import randrange

def quick_sort(collection: list) -> list:
    if len(collection) < 2:
        return collection
    
    idx = randrange(len(collection))
    p = collection.pop(idx)
    
    left = [x for x in collection if x <= p]
    right = [x for x in collection if x > p]
    
    return [*quick_sort(left), p, *quick_sort(right)]
```

**Negative (Original):**
```python
def quick_sort(collection: list) -> list:
    if len(collection) <= 1:
        return collection
    
    mid = len(collection) // 2
    pivot = collection[mid]
    
    left = []
    right = []
    middle = []
    
    for item in collection:
        if item < pivot:
            left.append(item)
        elif item > pivot:
            right.append(item)
        else:
            middle.append(item)
    
    return quick_sort(left) + middle + quick_sort(right)
```

### Two Sum (NeetCode)

**Positive (Plagiarized):**
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        
        for idx, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], idx]
            seen[num] = idx
```

**Negative (Original):**
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []
```

### Rotate List (NeetCode)

**Positive (Plagiarized):**
```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k == 0:
            return head
        
        start = head
        node, length = head, 0
        while node:
            node, length = node.next, length + 1
        
        if k % length == 0:
            return head
        
        k %= length
        left = right = head
        while right and right.next:
            if k <= 0:
                left = left.next
            right = right.next
            k -= 1
        
        tail, new_start, end = left, left.next, right
        tail.next, end.next = None, start
        
        return new_start
```

**Negative (Original):**
```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return head
        
        nodes = []
        curr = head
        while curr:
            nodes.append(curr)
            curr = curr.next
        
        n = len(nodes)
        k = k % n
        if k == 0:
            return head
        
        nodes[-1].next = nodes[0]
        nodes[n - k - 1].next = None
        
        return nodes[n - k]
```




## Links

Here are links:

- https://github.com/neetcode-gh/leetcode/tree/main/python
- https://github.com/TheAlgorithms/Python/tree/master/sorts

Then I implemented the 
