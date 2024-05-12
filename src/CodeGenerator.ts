import {
    AssignmentExpr,
    BinaryExpr,
    CallExpr,
    CompoundExpr,
    ConditionalDeclaration,
    ForLoopDeclaration,
    FunctionDeclaration,
    Identifier,
    LogicalExpr,
    MemberExpr,
    NumericLiteral,
    Program,
    Stmt,
    UnaryExpr,
    VarDeclaration
} from "./AST";
import { BufferDeclaration, NewStmt } from "./ExtraAST";

const CudaTypeToWGSL = {
    int: "i32",
    uint: "u32",
    float: "f32",
    void: ""
}

function getPrecedence(operator: string): number {
    const precedence = {
        "*": 3, "/": 3, "%": 3,
        "+": 2, "-": 2,
        "<": 1, "<=": 1, ">": 1, ">=": 1,
        "==": 0, "!=": 0,
        // add other operators and their precedence as necessary
    };
    return precedence[operator] || -1;  // return a low precedence for unknown operators
}

function needsParentheses(parentOp: string, childOp: string, isRight: boolean): boolean {
    const parentPrec = getPrecedence(parentOp);
    const childPrec = getPrecedence(childOp);

    if (childPrec < parentPrec) return true;
    // handle left-associativity for operators of the same precedence
    if (childPrec === parentPrec && isRight) return true;
    return false;
}

export function codeGenerator(node: NewStmt) {
    switch (node.kind) {
        case 'Program':
            const program = node as Program;
            // return program.body.map(codeGenerator).join('\n');

            const globalCode = program.globals.map(codeGenerator).join('\n');
            const bodyCode = program.body.map(codeGenerator).join('\n');
            return `${globalCode}\n${bodyCode}`;

        case 'Identifier':
            const identifier = node as Identifier;
            return identifier.symbol;

        case 'NumericLiteral':
            const numericLiteral = node as NumericLiteral;
            return numericLiteral.value;

        case "BinaryExpr": {
            const binaryExpression = node as BinaryExpr;
            let left = codeGenerator(binaryExpression.left);
            let right = codeGenerator(binaryExpression.right);

            if (binaryExpression.left.kind === "BinaryExpr" && 
                needsParentheses(binaryExpression.operator, (binaryExpression.left as BinaryExpr).operator, false)) {
                left = `(${left})`;
            }

            if (binaryExpression.right.kind === "BinaryExpr" &&
                needsParentheses(binaryExpression.operator, (binaryExpression.right as BinaryExpr).operator, true)) {
                right = `(${right})`;
            }

            return `${left} ${binaryExpression.operator} ${right}`;
        }

        case "RelationalExpr":
        case "EqualityExpr":
        case "LogicalExpr": {
            const binaryExpression = node as BinaryExpr;
            const left = codeGenerator(binaryExpression.left);
            const right = codeGenerator(binaryExpression.right);
            return `${left} ${binaryExpression.operator} ${right}`;
        }

        case 'UnaryExpr':
            const unaryExpr = node as UnaryExpr;
            return `${codeGenerator(unaryExpr.operand)}${unaryExpr.operator}`;

        case "VarDeclaration":
            const varDeclaration = node as VarDeclaration;
            const semicolon = varDeclaration.value && varDeclaration.value.kind !== "CallExpr" ? ";" : "";
            if (varDeclaration.value) {
                return `var ${varDeclaration.identifier}: ${CudaTypeToWGSL[varDeclaration.type]} = ${codeGenerator(varDeclaration.value)}${semicolon}`;
            }
            else {
                if (varDeclaration.pointer) return `${varDeclaration.identifier}: array<${CudaTypeToWGSL[varDeclaration.type]}>`;
                else return `${varDeclaration.identifier}: ${CudaTypeToWGSL[varDeclaration.type]}`;
            }


        case "CompoundExpr": {
            const compoundExpr = node as CompoundExpr;
            // Assignment expressions not always return a semicolon a = (b = 5) + (c = 3);
            const left = codeGenerator(compoundExpr.left);
            const right = codeGenerator(compoundExpr.right);
            return `${left} ${compoundExpr.operator} ${right};`;
        }

        case "AssignmentExpr": {
            const assignmentExpr = node as AssignmentExpr;
            // Assignment expressions not always return a semicolon a = (b = 5) + (c = 3);
            return `${codeGenerator(assignmentExpr.assigne)} = ${codeGenerator(assignmentExpr.value)};`;
        }

        case 'ForLoopDeclaration': {
            const forLoopDeclaration = node as ForLoopDeclaration;
            const init = `${codeGenerator(forLoopDeclaration.init)}`;
            const condition = `${codeGenerator(forLoopDeclaration.condition)}`;
            const increment = `${codeGenerator(forLoopDeclaration.increment)}`;
            const body = `${forLoopDeclaration.body.map(codeGenerator).join('\n')}`;
            return `for (${init} ${condition}; ${increment}) {\n${body}\n}`
        }

        case "ConditionalDeclaration": {
            const conditionalDeclaration = node as ConditionalDeclaration;
            const test = codeGenerator(conditionalDeclaration.test);
            const body = conditionalDeclaration.body.map(codeGenerator).join('\n');
            return `if (${test}) {\n${body}\n}`;
        }

        case "MemberExpr":
            const memberExpr = node as MemberExpr;
            if (memberExpr.computed) return `${codeGenerator(memberExpr.object)}[${codeGenerator(memberExpr.property)}]`;
            else return `${codeGenerator(memberExpr.object)}.${codeGenerator(memberExpr.property)}`;

        case "CallExpr":
            const callExpr = node as CallExpr;
            return `${codeGenerator(callExpr.caller)}(${callExpr.args.map(codeGenerator).join(', ')});`;

        case "FunctionDeclaration": {
            const functionDeclaration = node as FunctionDeclaration;
            let body = functionDeclaration.body.map(codeGenerator).join('\n');
            if (functionDeclaration.body.length > 0) body = "\n" + body + "\n";

            let returnType = "";
            if (functionDeclaration.type !== "void") returnType = ` -> ${CudaTypeToWGSL[functionDeclaration.type]}`;

            const args = functionDeclaration.parameters.map(codeGenerator).join(', ');
            return `fn ${functionDeclaration.name}(${args})${returnType} {${body}}`;
        }

        case "BufferDeclaration": {
            const bufferDeclaration = node as BufferDeclaration;
            const group = `@group(%group_${bufferDeclaration.bufferCounter}%)`;
            const binding = `@binding(%binding_${bufferDeclaration.bufferCounter}%)`;

            const param = bufferDeclaration.value as VarDeclaration;
            const paramStr = param.pointer ? `array<${CudaTypeToWGSL[param.type]}, %array_size_${bufferDeclaration.bufferCounter}%>` : `${CudaTypeToWGSL[param.type]}`;
            return `${group} ${binding} var<${bufferDeclaration.storage}, ${bufferDeclaration.access}> ${bufferDeclaration.identifier}: ${paramStr};`;
        }

        default:
            console.warn("[CodeGenerator] Node handler not found", node);
            throw new Error(`Cannot handle ${node.kind}`);
    }
}