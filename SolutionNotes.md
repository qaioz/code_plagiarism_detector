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

Then I implemented the chunking using the library I found: treesitter. 

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

I was going to finish it and max it out. 
I then decided it does not worth my time and I would rather 
But dang, it, I will do it anyway, I will finish because I started 


## Links

Here are links:

- https://github.com/neetcode-gh/leetcode/tree/main/python
- https://github.com/TheAlgorithms/Python/tree/master/sorts

Then I implemented the 
