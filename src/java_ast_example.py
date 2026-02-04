import asyncio
from preprocessing.java_ast_async import extract_methods_async, extract_classes_async


def process(result):
    # Some function aggregation combination
    pass
    

async def extract_methods():
    # Async extract functions from files
    tasks = [
        asyncio.create_task(extract_methods_async("some/path/A.java")),
        asyncio.create_task(extract_methods_async("some/path/B.java")),
        asyncio.create_task(extract_methods_async("some/path/C.java")),
    ]

    # Process each completed element streamwise
    for completed in asyncio.as_completed(tasks):
        result = await completed
        process(result)


async def extract_classes():
    # Async extract classes from files
    tasks = [
        asyncio.create_task(extract_classes_async("some/path/A.java")),
        asyncio.create_task(extract_classes_async("some/path/B.java")),
        asyncio.create_task(extract_classes_async("some/path/C.java")),
    ] 

    # Process each completed element streamwise
    for completed in asyncio.as_completed(tasks):
        result = await completed
        process(result)


if __name__ == '__main__':
    asyncio.run(main())
