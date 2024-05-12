import { traverser } from "./Traverser";
import {
    Program,
    Stmt,
    BinaryExpr,
    VarDeclaration,
    MemberExpr,
    Identifier,
    CallExpr,
    FunctionDeclaration,
    RelationalExpr,
    UnaryExpr,
    ForLoopDeclaration,
    AssignmentExpr,
    ConditionalDeclaration,
    LogicalExpr,
    CompoundExpr
} from "./AST";
import { BufferDeclaration, NewStmt } from "./ExtraAST";

export function transformer(ast: Stmt) {
    let newAst: Program = {
        kind: "Program",
        body: [],
        globals: [],
        counters: {
            buffer: 0,
        }
    };
    // ast._context = newAst.body;
    ast._context = {
        body: newAst.body,
        globals: newAst.globals,
        counters: newAst.counters
    };
    traverser(ast, {
        Program: {
            enter(node: Program, parent) {
            }
        },
        NumericLiteral: {
            // We'll visit them on enter.
            enter(node, parent) {
                // parent._context.push({
                //     kind: "NumericLiteral",
                //     value: node.value
                // });


                // node._context = {
                //     body: [node.object, node.property],
                //     globals: []
                // }
                parent._context?.body.push({
                    kind: "NumericLiteral",
                    value: node.value
                });
            }
        },

        BinaryExpr: {
            enter(node: BinaryExpr, parent: Stmt) {
                // node._context = [node.left, node.right];
                // parent._context.push(node);

                node._context = {body: [node.left, node.right], globals: [] }
                parent._context?.body.push(node);
            }
        },
        RelationalExpr: {
            enter(node: RelationalExpr, parent: Stmt) {
                // node._context = [node.left, node.right];
                // parent._context.push(node);
                node._context = {body: [node.left, node.right], globals: [] }
                parent._context?.body.push(node);
            }
        },
        EqualityExpr: {
            enter(node: RelationalExpr, parent: Stmt) {
                // node._context = [node.left, node.right];
                // parent._context.push(node);
                node._context = {body: [node.left, node.right], globals: [] }
                parent._context?.body.push(node);
                
            }
        },

        LogicalExpr: {
            enter(node: LogicalExpr, parent: Stmt) {
                // node._context = [node.left, node.right];
                // parent._context.push(node);
                node._context = {body: [node.left, node.right], globals: [] }
                parent._context?.body.push(node);
            }
        },

        UnaryExpr: {
            enter(node: UnaryExpr, parent: Stmt) {
                // node._context = [node.operand];
                // parent._context.push(node);
                node._context = {body: [node.operand], globals: [] }
                parent._context?.body.push(node);
            }
        },

        CompoundExpr: {
            enter(node: CompoundExpr, parent: Stmt) {
                // node._context = [node.left, node.right];
                // parent._context.push(node);
                node._context = {body: [node.left, node.right], globals: [] }
                parent._context?.body.push(node);
            }
        },

        AssignmentExpr: {
            enter(node: AssignmentExpr, parent: Stmt) {
                // node._context = [node.assigne, node.value];
                // parent._context.push(node);
                node._context = {body: [node.assigne, node.value], globals: [] }
                parent._context?.body.push(node);
            }
        },

        ForLoopDeclaration: {
            enter(node: ForLoopDeclaration, parent: Stmt) {
                // node._context = [node.init, node.condition, node.increment, ...node.body];
                // parent._context.push(node);
                node._context = {body: [node.init, node.condition, node.increment, ...node.body], globals: [] }
                parent._context?.body.push(node);
            }
        },

        ConditionalDeclaration: {
            enter(node: ConditionalDeclaration, parent: Stmt) {
                node._context = {
                    body: [node.test, ...node.body],
                    globals: []
                }
                parent._context?.body.push(node);
            }
        },

        VarDeclaration: {
            enter(node: VarDeclaration, parent: Stmt) {
                node._context = {body: [node.value], globals: [] }
                parent._context?.body.push(node);
            }
        },

        MemberExpr: {
            enter(node: MemberExpr, parent: Stmt) {
                // node._context = [node.object, node.property];
                // parent._context.push(node);

                node._context = {
                    body: [node.object, node.property],
                    globals: []
                }
                parent._context?.body.push(node);
                // parent._context?.globals.push(...globals);
            }
        },

        Identifier: {
            enter(node: Identifier, parent: Stmt) {
                // parent._context.push(node);

                // node._context = {
                //     body: [node.object, node.property],
                //     globals: []
                // }
                parent._context?.body.push(node);
            }
        },

        CallExpr: {
            enter(node: CallExpr, parent: Stmt) {
                const expression: CallExpr = {
                    kind: "CallExpr",
                    args: [],
                    caller: node.caller
                }
                // node._context = expression.args;
                // parent._context.push(expression);

                node._context = {body: expression.args, globals: [] }
                parent._context?.body.push(expression);
            }
        },

        BufferDeclaration: {
            enter(node: FunctionDeclaration, parent: Stmt) {
            }
        },

        FunctionDeclaration: {
            enter(node: FunctionDeclaration, parent: Stmt) {
                const func: FunctionDeclaration = {
                    kind: "FunctionDeclaration",
                    name: node.name,
                    qualifiers: node.qualifiers,
                    type: node.type,
                    body: node.body,
                    parameters: []
                }

                const body = [...func.body];
                // Normal function
                if (!node.qualifiers.includes("__global__")) {
                    func.parameters.push(...node.parameters);
                    body.push(...func.parameters);
                }

                // Global function is entrypoint, needs BufferDeclaration
                const globals: NewStmt[] = [];
                if (node.qualifiers.includes("__global__")) {
                    for (let param of node.parameters) {
                        const bufferDeclaration: BufferDeclaration = {
                            kind: "BufferDeclaration",
                            bufferCounter: parent._context.counters.buffer++,
                            storage: "storage",
                            access: "read_write",
                            identifier: param.identifier,
                            value: param
                        }
                        globals.push(bufferDeclaration);
                    }
                }

                node._context = {
                    body: body,
                    globals: globals
                }
                parent._context?.globals.push(...globals);
                parent._context?.body.push(func);
            }
        }
    });
    return newAst;
}