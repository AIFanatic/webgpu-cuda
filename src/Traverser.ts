import {
    AssignmentExpr,
    BinaryExpr,
    CallExpr,
    ConditionalDeclaration,
    ForLoopDeclaration,
    FunctionDeclaration,
    MemberExpr,
    Program,
    Stmt,
    UnaryExpr,
    VarDeclaration
} from "./AST";

export function traverser(ast, visitor) {
    function traverseArray(array, parent) {
        array.forEach((child) => {
            traverseNode(child, parent);
        });
    }
    function traverseNode(node: Stmt, parent) {
        let methods = visitor[node.kind];

        if (methods && methods.enter) methods.enter(node, parent);
        else {
            console.warn(node);
            throw Error("Enter method not defined");
        }
        switch (node.kind) {
            case "Program":
                const program = node as Program;
                traverseArray(program.body, node);
                break;

            case "UnaryExpr":
                const unaryExpr = node as UnaryExpr;
                traverseNode(unaryExpr.operand, node);
                break;

            case "EqualityExpr":
            case "LogicalExpr":
            case "RelationalExpr":
            case "CompoundExpr":
            case "BinaryExpr":
                const binaryExpr = node as BinaryExpr;
                traverseNode(binaryExpr.left, node);
                traverseNode(binaryExpr.right, node);
                break;
            case "CallExpr":
                const callExpr = node as CallExpr;
                traverseArray(callExpr.args, node);
                break;

            case "AssignmentExpr":
                const assignmentExpr = node as AssignmentExpr;
                traverseNode(assignmentExpr.assigne, node);
                traverseNode(assignmentExpr.value, node);
                break;
            
            case "NumericLiteral":
            case "Identifier":
                break;

            case "ForLoopDeclaration":
                const forLoopDeclaration = node as ForLoopDeclaration;
                traverseNode(forLoopDeclaration.init, node);
                traverseNode(forLoopDeclaration.condition, node);
                traverseNode(forLoopDeclaration.increment, node);
                traverseArray(forLoopDeclaration.body, node);
                break;

            case "ConditionalDeclaration":
                const conditionalDeclaration = node as ConditionalDeclaration;
                traverseNode(conditionalDeclaration.test, node);
                traverseArray(conditionalDeclaration.body, node);
                break;

            case "VarDeclaration":
                const varDeclaration = node as VarDeclaration;
                if (varDeclaration.value) traverseNode(varDeclaration.value, node);
                break;

            case "MemberExpr":
                const memberExpr = node as MemberExpr;
                traverseNode(memberExpr.object, node);
                traverseNode(memberExpr.property, node);
                break;

            case "FunctionDeclaration":
                const functionDeclaration = node as FunctionDeclaration;
                traverseArray(functionDeclaration.parameters, node);
                traverseArray(functionDeclaration.body, node);
                break;

            case "BufferDeclaration":
                console.log('case "BufferDeclaration":', node)
                throw Error("ERGERGERG")
            default:
                console.warn("[Traverser] Error processing node", node);
                throw new Error(`Cannot handle ${node.kind}`);
        }
        if (methods && methods.exit) {
            methods.exit(node, parent);
        }
    }
    traverseNode(ast, null);
}