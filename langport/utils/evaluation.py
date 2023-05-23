# license: MIT (C) tardyp
import ast
import unittest


def safe_eval(expr, variables=None):
    """
    Safely evaluate a a string containing a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, numbers, tuples, lists, dicts, booleans,
    and None. safe operators are allowed (and, or, ==, !=, not, +, -, ^, %, in, is)
    """
    _safe_names = {'None': None, 'True': True, 'False': False}
    _safe_nodes = [
        'Add', 'And', 'BinOp', 'BitAnd', 'BitOr', 'BitXor', 'BoolOp',
        'Compare', 'Dict', 'Eq', 'Expr', 'Expression', 'For',
        'Gt', 'GtE', 'Is', 'In', 'IsNot', 'LShift', 'List',
        'Load', 'Lt', 'LtE', 'Mod', 'Name', 'Not', 'NotEq', 'NotIn',
        'Num', 'Or', 'RShift', 'Set', 'Slice', 'Str', 'Sub',
        'Tuple', 'UAdd', 'USub', 'UnaryOp', 'boolop', 'cmpop',
        'expr', 'expr_context', 'operator', 'slice', 'unaryop', 'Constant', 'Div']
    node = ast.parse(expr, mode='eval')
    for subnode in ast.walk(node):
        subnode_name = type(subnode).__name__
        if isinstance(subnode, ast.Name):
            if subnode.id not in _safe_names and subnode.id not in variables:
                raise ValueError("Unsafe expression node {}. contains {}".format(expr, subnode.id))
        if subnode_name not in _safe_nodes:
            raise ValueError("Unsafe expression name {}. contains {}".format(expr, subnode_name))

    return eval(expr, variables)



class SafeEvalTests(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(safe_eval("1", {}), 1)

    def test_local(self):
        self.assertEqual(safe_eval("a", {'a': 2}), 2)

    def test_local_bool(self):
        self.assertEqual(safe_eval("a==2", {'a': 2}), True)

    def test_lambda(self):
        self.assertRaises(ValueError, safe_eval, "lambda : None", {'a': 2})

    def test_bad_name(self):
        self.assertRaises(ValueError, safe_eval, "a == None2", {'a': 2})

    def test_attr(self):
        self.assertRaises(ValueError, safe_eval, "a.__dict__", {'a': 2})

    def test_eval(self):
        self.assertRaises(ValueError, safe_eval, "eval('os.exit()')", {})

    def test_exec(self):
        self.assertRaises(SyntaxError, safe_eval, "exec 'import os'", {})

    def test_multiply(self):
        self.assertRaises(ValueError, safe_eval, "'s' * 3", {})

    def test_power(self):
        self.assertRaises(ValueError, safe_eval, "3 ** 3", {})

    def test_comprehensions(self):
        self.assertRaises(ValueError, safe_eval, "[i for i in [1,2]]", {'i': 1})