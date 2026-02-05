"""
Extracts content from files and returns a list of either classes or methods
"""
import asyncio
import javalang  # pylint: disable=import-error

def _parse_and_extract(source: str):
    """
    Blocking function: parse Java file and extract methods.
    Runs in executor.
    """
    tree = javalang.parse.parse(source)

    results = []
    for _, node in tree.filter(
        (javalang.tree.MethodDeclaration,
         javalang.tree.ConstructorDeclaration)
    ):
        name = node.name if hasattr(node, "name") else "<constructor>"
        results.append((name, node))

    return results


async def extract_methods_async(java_file: str):
    """
    Async wrapper around javalang parsing.

    @param java_path: Path to a .java file
    @return: List of (method_name, ast_node)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _parse_and_extract, java_file)


def _parse_and_extract_classes(source: str):
    """
    Blocking function: parse Java file and extract class-like declarations.
    Runs in executor.
    """
    tree = javalang.parse.parse(source)

    results = []

    for _, node in tree.filter((
        javalang.tree.ClassDeclaration,
        javalang.tree.InterfaceDeclaration,
        javalang.tree.EnumDeclaration,
    )):
        results.append((node.name, node))

    return results


async def extract_classes_async(java_file: str):
    """
    Async wrapper around javalang parsing for class-level declarations.

    @param java_file: the content of a java file.
    @return: List of (class_name, ast_node)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _parse_and_extract_classes, java_file)
