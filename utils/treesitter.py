from tree_sitter import Node


def get_tokens_from_node(node: Node):

    tokens = []

    def traverse(node):
        if node.child_count == 0:  # It's a terminal node
            if node.type == 'identifier':
                tokens.append('<ID>')
            elif node.type == 'comment':
                # do nothing
                pass
            elif node.type == 'number_literal':
                tokens.append('<NUM>')
            elif node.type == 'escape_sequence':
                tokens.append('<ESCAPE>')
            elif node.type == 'statement_identifier':
                tokens.append('<LAB>')
            else:
                tokens.append(node.text.decode('utf-8'))
        else:
            for child in node.children:
                traverse(child)

    traverse(node)
    return tokens
