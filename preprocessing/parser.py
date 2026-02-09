"""
Parser module
"""
import javalang

def parse_java(code: str):
    """
    Extract AST tree from entire java file
    """
    return javalang.parse.parse(code)

def extract_method_decls(code: str):
    """
    Extract method decalations
    """
    tree = parse_java(code)
    return {
        node.name: node
        for _, node in tree.filter(javalang.tree.MethodDeclaration)
    }
