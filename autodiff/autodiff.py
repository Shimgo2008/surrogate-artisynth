from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto


import numpy as np


def as_node(value):
    if isinstance(value, Node):
        return value
    else:
        return Node(value)


class OpKind(Enum):
    INFIX = auto()
    PREFIX = auto()
    FUNCTION = auto()


class Op(Enum):
    ADD = ("+",   10, OpKind.INFIX)
    SUB = ("-",   10, OpKind.INFIX)
    MUL = ("*",   20, OpKind.INFIX)
    DIV = ("/",   20, OpKind.INFIX)
    POW = ("**",  30, OpKind.INFIX)

    NEG = ("-",   40, OpKind.PREFIX)

    GT = (">",   5, OpKind.INFIX)
    GE = (">=",  5, OpKind.INFIX)
    LT = ("<",   5, OpKind.INFIX)
    LE = ("<=",  5, OpKind.INFIX)

    EXP = ("exp", 50, OpKind.FUNCTION)
    SIN = ("sin", 50, OpKind.FUNCTION)
    COS = ("cos", 50, OpKind.FUNCTION)
    LOG = ("log", 50, OpKind.FUNCTION)
    MEAN = ("mean", 50, OpKind.FUNCTION)

    def __init__(self, symbol, precedence, kind):
        self.symbol = symbol
        self.precedence = precedence
        self.kind = kind


class D:
    """
    D[k](f)(Node)
    """
    def __class_getitem__(cls, n):
        return lambda f: lambda x: D.nth_derivative(f, x, n)

    def nth_derivative(func, x, n):
        x = as_node(x)
        y = func(x)

        for _ in range(n):
            y = y.diff(x)

        return y


class ComputeLayer:
    def __init__(self, func, grad_func, *, op=None, op_data=None):
        self.func = func
        self.grad_func = grad_func
        self.op = op
        self.op_data = op_data

    def __call__(self, *args):
        args = tuple(as_node(arg) for arg in args)
        values = tuple(arg.data for arg in args)

        val = self.func(*values)
        out = Node(val, parents=args, op=self.op, op_data=self.op_data)

        def _backward(grad):
            return self.grad_func(grad, *args)

        out._backward = _backward

        return out


class Node:
    def __init__(self, value, parents=(), backward= None, *,
        op=None,
        name=None,
        op_data=None
    ):
        self.data = np.asarray(value)
        self.grad = np.zeros_like(self.data, dtype=float)
        self.parents: tuple[Node] = tuple(parents)
        self._backward: callable = backward or (lambda grad: ())

        # 表示専用
        self.op = op
        self.name = name
        self.op_data = op_data

    def backward(self, grad=None) -> None:
        topo_order = self._topo()

        # 前回の勾配をリセット
        for node in topo_order:
            node.grad = np.zeros_like(node.data, dtype=float)

        if grad is None:
            grad = np.ones_like(self.data, dtype=float)

        self.grad = grad

        for node in reversed(topo_order):
            node_grad = as_node(node.grad)
            parent_grads = node._backward(node_grad)

            for parent, parent_grad in zip(node.parents, parent_grads):
                grad_data = self._sum_to_shape(
                    parent_grad.data,
                    parent.data.shape
                )
                parent.grad += grad_data

    def diff(self, wrt: 'Node', grad=1.0) -> 'Node':
        topo_order = self._topo()

        grads = {
            self: as_node(grad)
        }

        for node in reversed(topo_order):
            node_grad = grads[node]
            parent_grads = node._backward(node_grad)

            for parent, parent_grad in zip(node.parents, parent_grads):
                if parent in grads:
                    grads[parent] = grads[parent] + parent_grad
                else:
                    grads[parent] = parent_grad

        return grads.get(wrt, Node(0.0))

    def _topo(self) -> list['Node']:
        visited = set()
        topo_order = []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                topo_order.append(node)

        dfs(self)
        return topo_order

    def __add__(self, other):
        return ComputeLayer(
            func=lambda a, b: a + b,
            grad_func=lambda grad, a, b: (
                grad,
                grad,
            ),
            op=Op.ADD,
        )(self, other)

    def __sub__(self, other):
        return ComputeLayer(
            func=lambda a, b: a - b,
            grad_func=lambda grad, a, b: (
                grad,
                -grad,
            ),
            op=Op.SUB,
        )(self, other)

    def __mul__(self, other):
        return ComputeLayer(
            func=lambda a, b: a * b,
            grad_func=lambda grad, a, b: (
                grad * b,
                grad * a,
            ),
            op=Op.MUL,
        )(self, other)

    def __truediv__(self, other):
        return ComputeLayer(
            func=lambda a, b: a / b,
            grad_func=lambda grad, a, b: (
                grad / b,
                -grad * a / b**2,
            ),
            op=Op.DIV,
        )(self, other)

    def __pow__(self, exponent):
        return ComputeLayer(
            func=lambda a: a ** exponent,
            grad_func=lambda grad, a: (
                grad * exponent * a ** (exponent - 1),
            ),
            op=Op.POW,
            op_data=exponent,
        )(self)

    def __matmul__(self, other):
        return ComputeLayer(
            func=lambda a, b: a @ b,
            grad_func=lambda grad, a, b: (
                grad @ as_node(b.data.T),
                as_node(a.data.T) @ grad,
            ),
            op=Op.MUL,
        )(self, other)

    def __neg__(self):
        return ComputeLayer(
            func=lambda a: -a,
            grad_func=lambda grad, a: (
                -grad,
            ),
            op=Op.NEG,
        )(self)

    def __radd__(self, other):
        return as_node(other) + self

    def __rsub__(self, other):
        return as_node(other) - self

    def __rmul__(self, other):
        return as_node(other) * self

    def __rtruediv__(self, other):
        return as_node(other) / self

    def __le__(self, other):
        return ComputeLayer(
            func=lambda a, b: a <= b,
            grad_func=lambda grad, a, b: (
                grad,
                grad,
            ),
            op=Op.LE,
        )(self, other)

    def __lt__(self, other):
        return ComputeLayer(
            func=lambda a, b: a < b,
            grad_func=lambda grad, a, b: (
                grad,
                grad,
            ),
            op=Op.LT,
        )(self, other)

    def __ge__(self, other):
        return ComputeLayer(
            func=lambda a, b: a >= b,
            grad_func=lambda grad, a, b: (
                grad,
                grad,
            ),
            op=Op.GE,
        )(self, other)

    def __gt__(self, other):
        return ComputeLayer(
            func=lambda a, b: a > b,
            grad_func=lambda grad, a, b: (
                grad,
                grad,
            ),
            op=Op.GT,
        )(self, other)

    def exp(self):
        return ComputeLayer(
            func=np.exp,
            grad_func=lambda grad, a: (
                grad * np.exp(a),
            ),
            op=Op.EXP,
        )(self)

    def log(self):
        return ComputeLayer(
            func=np.log,
            grad_func=lambda grad, a: (
                grad / a,
            ),
            op=Op.LOG,
        )(self)

    def mean(self):
        return ComputeLayer(
            func=np.mean,
            grad_func=lambda grad, a: (
                grad * np.ones_like(a.data) / a.data.size,
            ),
            op=Op.MEAN,
            op_data="mean",
        )(self)

    def ReLU(self):
        return ComputeLayer(
            func=lambda a: np.maximum(0, a),
            grad_func=lambda grad, a: (
                grad * (a > 0).data.astype(float),
            ),
            op=OpKind.FUNCTION,
            op_data="ReLU",
        )(self)

    def _sum_to_shape(self, grad: np.ndarray, shape: tuple[int, ...]):
        grad = np.asarray(grad)

        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, original_size in enumerate(shape):
            if original_size == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def _expr(self, parent_op=None, side=None):
        """Render the graph with only presentation-level cleanup applied."""
        display_node = self._display_node()
        return display_node._render_expr(parent_op=parent_op, side=side)

    def _render_expr(self, parent_op=None, side=None):

        if self.name is not None:
            return self.name

        if self.op is None:
            return self._literal_expr()

        if self.op.kind is OpKind.FUNCTION:
            inner = self.parents[0]._render_expr()
            expr = f"{self.op.symbol}({inner})"

        elif self.op.kind is OpKind.PREFIX:
            child = self.parents[0]

            inner = child._render_expr(
                parent_op=self.op,
                side="right",
            )

            expr = f"{self.op.symbol}{inner}"

        elif self.op is Op.POW:
            base = self.parents[0]._render_expr(
                parent_op=self.op,
                side="left",
            )

            exponent = (
                self.op_data._render_expr()
                if isinstance(self.op_data, Node)
                else repr(self.op_data)
            )

            expr = f"{base} {self.op.symbol} {exponent}"

        else:
            left_node, right_node = self.parents

            left = left_node._render_expr(
                parent_op=self.op,
                side="left",
            )

            right = right_node._render_expr(
                parent_op=self.op,
                side="right",
            )

            expr = f"{left} {self.op.symbol} {right}"

        if parent_op is None:
            return expr

        if self._needs_parentheses(parent_op, side):
            return f"({expr})"

        return expr

    def _literal_expr(self):
        value = self.data
        if np.ndim(value) == 0 and np.issubdtype(np.asarray(value).dtype, np.floating):
            if value == 0:
                value = abs(value)
        if np.ndim(value) == 0 and np.iscomplexobj(value):
            value = complex(value)
            real = 0.0 if value.real == 0 else value.real
            imag = 0.0 if value.imag == 0 else value.imag
            if imag == 0:
                return repr(real)
            return repr(complex(real, imag))
        return repr(value)

    def _display_node(self):
        if self.op is None:
            return self

        if self.op is Op.POW:
            base = self.parents[0]._display_node()
            exponent = (
                self.op_data._display_node()
                if isinstance(self.op_data, Node)
                else self.op_data
            )
            if base is self.parents[0] and exponent is self.op_data:
                return self
            return Node(self.data, parents=(base,), op=self.op, op_data=exponent)

        children = tuple(parent._display_node() for parent in self.parents)

        # Keep repr() readable by folding only obvious function constants.
        if self.op is Op.EXP:
            child = children[0]
            if child._is_literal_value(0):
                return Node(1)

        if self.op is Op.NEG:
            child = children[0]
            if child._is_literal():
                try:
                    return Node(-child.data)
                except (TypeError, ValueError):
                    pass

        if self.op is Op.ADD:
            left, right = children
            if left.op is Op.NEG and left.parents[0]._same_expression(right):
                return Node(0)
            if right.op is Op.NEG and right.parents[0]._same_expression(left):
                return Node(0)
            if left._is_literal() and right._is_literal():
                try:
                    return Node(left.data + right.data)
                except (TypeError, ValueError):
                    pass

        if self.op is Op.SUB:
            left, right = children
            if left._same_expression(right):
                return Node(0)
            if right._is_const(0):
                return left

        if self.op is Op.MUL:
            left, right = children
            if left._is_const(0):
                return left
            if right._is_const(0):
                return right
            if left._is_const(1):
                return right
            if right._is_const(1):
                return left
            if left._is_literal() and right._is_literal():
                try:
                    return Node(left.data * right.data)
                except (TypeError, ValueError):
                    pass

        if self.op is Op.DIV:
            left, right = children
            if left._is_const(0):
                return left
            if right._is_const(1):
                return left
            if (
                left._is_literal()
                and right._is_literal()
                and np.iscomplexobj(right.data)
            ):
                try:
                    return Node(left.data / right.data)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

        if children == self.parents:
            return self
        return Node(
            self.data,
            parents=children,
            op=self.op,
            op_data=self.op_data,
        )

    def _simplified_node(self):
        if len(self.parents) != 2:
            return self

        left, right = self.parents

        if self.op is Op.ADD:
            # x + 0 -> x
            if right._is_const(0):
                return left

            # 0 + x -> x
            if left._is_const(0):
                return right

        elif self.op is Op.SUB:
            if left._same_expression(right):
                return Node(0)

            if right._is_const(0):
                return left

        elif self.op is Op.MUL:
            # x * 0 -> 0
            if right._is_const(0):
                return right

            # 0 * x -> 0
            if left._is_const(0):
                return left

            # x * 1 -> x
            if right._is_const(1):
                return left

            # 1 * x -> x
            if left._is_const(1):
                return right

        elif self.op is Op.DIV:
            # x / 1 -> x
            if right._is_const(1):
                return left

        return self

    def _is_literal(self) -> bool:
        return self.op is None and self.name is None

    def _is_literal_value(self, value) -> bool:
        if not self._is_literal():
            return False

        try:
            return bool(np.all(np.asarray(self.data) == value))
        except (TypeError, ValueError):
            return False

    def _is_const(self, value) -> bool:
        if not self._is_literal():
            return False

        try:
            return bool(np.all(np.asarray(self.data) == value))
        except (TypeError, ValueError):
            return False

    def _same_expression(self, other: "Node") -> bool:
        if self is other:
            return True

        if not isinstance(other, Node):
            return False

        if self.op is None or other.op is None:
            if not (self._is_literal() and other._is_literal()):
                return False
            try:
                return bool(np.array_equal(self.data, other.data))
            except (TypeError, ValueError):
                return False

        if self.op is not other.op or len(self.parents) != len(other.parents):
            return False

        if self.op is Op.POW:
            left_exponent = self.op_data
            right_exponent = other.op_data
            if isinstance(left_exponent, Node) and isinstance(right_exponent, Node):
                if not left_exponent._same_expression(right_exponent):
                    return False
            elif isinstance(left_exponent, Node) or isinstance(right_exponent, Node):
                return False
            else:
                try:
                    if not np.array_equal(left_exponent, right_exponent):
                        return False
                except (TypeError, ValueError):
                    return False

        return all(
            left._same_expression(right)
            for left, right in zip(self.parents, other.parents)
        )

    def simplify(self) -> "Node":
        # leaf
        if self.op is None:
            return self

        # f(x)
        if self.op.kind is OpKind.FUNCTION:
            child = self.parents[0].simplify()

            # exp(0) -> 1
            if self.op is Op.EXP and child._is_literal_value(0):
                return Node(1)


            # 必要になったらここに
            #
            # exp(0) -> 1
            # log(1) -> 0
            # sin(0) -> 0
            # cos(0) -> 1
            #
            # などを追加

            # 子が変わっていなければ元ノードを再利用
            if child is self.parents[0]:
                return self

            # 現状 exp だけならこれでOK
            if self.op is Op.EXP:
                return child.exp()

            raise NotImplementedError(
                f"simplify rebuild is not implemented for {self.op}"
            )

        # -x
        if self.op.kind is OpKind.PREFIX:
            child = self.parents[0].simplify()

            # -constant -> constant
            if child._is_literal():
                return Node(-child.data)

            if child is self.parents[0]:
                return self

            return -child

        # x ** a
        if self.op is Op.POW:
            base = self.parents[0].simplify()

            exponent = (
                self.op_data.simplify()
                if isinstance(self.op_data, Node)
                else self.op_data
            )

            # exponent が literal Node なら値を取り出す
            exponent_value = (
                exponent.data
                if isinstance(exponent, Node) and exponent._is_literal()
                else exponent
            )

            # x ** 0 -> 1
            if not isinstance(exponent_value, Node):
                try:
                    if np.all(np.asarray(exponent_value) == 0):
                        return Node(1)
                except (TypeError, ValueError):
                    pass

            # x ** 1 -> x
            if not isinstance(exponent_value, Node):
                try:
                    if np.all(np.asarray(exponent_value) == 1):
                        return base
                except (TypeError, ValueError):
                    pass

            # constant ** constant
            if base._is_literal() and not isinstance(exponent_value, Node):
                try:
                    return Node(base.data ** exponent_value)
                except (TypeError, ValueError):
                    pass

            if (
                base is self.parents[0]
                and exponent is self.op_data
            ):
                return self

            return base ** exponent

        # binary operation
        left = self.parents[0].simplify()
        right = self.parents[1].simplify()

        #
        # 恒等変形
        #

        if self.op is Op.ADD:
            # 0 + x -> x
            if left._is_const(0):
                return right

            # x + 0 -> x
            if right._is_const(0):
                return left

        elif self.op is Op.SUB:
            # x - 0 -> x
            if right._is_const(0):
                return left

            # 同じ Node なら x - x -> 0
            if left._same_expression(right):
                return Node(0)

        elif self.op is Op.MUL:
            # 0 * x -> 0
            if left._is_const(0):
                return left

            # x * 0 -> 0
            if right._is_const(0):
                return right

            # 1 * x -> x
            if left._is_const(1):
                return right

            # x * 1 -> x
            if right._is_const(1):
                return left

        elif self.op is Op.DIV:
            # x / 1 -> x
            if right._is_const(1):
                return left

        #
        # constant folding
        #
        # 2 + 3   -> 5
        # 1j * 0  -> 0j
        # 1 - 1   -> 0
        # 2 * 1j  -> 2j
        #

        if left._is_literal() and right._is_literal():
            if self.op is Op.ADD:
                return Node(left.data + right.data)

            if self.op is Op.SUB:
                return Node(left.data - right.data)

            if self.op is Op.MUL:
                return Node(left.data * right.data)

            if self.op is Op.DIV:
                return Node(left.data / right.data)

        #
        # 何も簡約できなかった
        #

        if left is self.parents[0] and right is self.parents[1]:
            return self

        # 簡約された children でグラフを作り直す
        if self.op is Op.ADD:
            return left + right

        if self.op is Op.SUB:
            return left - right

        if self.op is Op.MUL:
            return left * right

        if self.op is Op.DIV:
            return left / right

        raise NotImplementedError(
            f"simplify rebuild is not implemented for {self.op}"
        )

    def _needs_parentheses(self, parent_op, side):
        my_prec = self.op.precedence
        parent_prec = parent_op.precedence

        # x * (a + b)
        if my_prec < parent_prec:
            return True

        # 優先順位が高ければ不要
        if my_prec > parent_prec:
            return False

        # 同順位で非結合的な右辺
        #
        # a - (b + c)
        # a - (b - c)
        if parent_op is Op.SUB and side == "right":
            return True

        # a / (b * c)
        # a / (b / c)
        if parent_op is Op.DIV and side == "right":
            return True

        # (a ** b) ** c
        if (
            parent_op is Op.POW
            and self.op is Op.POW
            and side == "left"
        ):
            return True

        return False

    def __repr__(self):
        return self._expr()

    def __str__(self):
        return self.simplify()._expr()

    def __format__(self, format_spec):
        return self.data.__format__(format_spec)

    def debug(self):
        return (
            f"Node("
            f"expr={self._expr()!r}, "
            f"data={self.data!r}"
            f")"
        )
