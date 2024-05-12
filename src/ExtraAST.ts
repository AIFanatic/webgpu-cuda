import { Expr, NodeType, VarDeclaration } from "./AST";

export type ExtraNodeType = NodeType 
    | "BufferDeclaration"
    | "ArrayType"
;

// Define a base statement type with a generic that can be a discriminated union
export interface Stmt<K extends NodeType | ExtraNodeType> {
    kind: K;
}

export interface NewStmt extends Stmt<NodeType | ExtraNodeType> {}

export interface ArrayType extends NewStmt {
    kind: "ArrayType";
    type: string;
    size: string;
}

export interface BufferDeclaration extends NewStmt {
    kind: "BufferDeclaration";
    bufferCounter: number;
    storage: "storage";
    access: "read_write";
    identifier: string;
    value: Expr;
}