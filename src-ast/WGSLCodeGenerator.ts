import { Token, TokenType, tokenize } from "./Lexer";

const TypeToWGSL = {
    uint: "u32",
    int: "i32",
    float: "f32",
};

export class WGSLCodeGenerator {
    private tokens: Token[];
    private index: number;
    
    constructor() {}
    
    private eat(): Token { return this.tokens[++this.index]; }
    private current(): Token { return this.tokens[this.index]; }
    private previous(offset = 1): Token { return this.tokens[this.index - offset]; }
    private next(offset = 1): Token { return this.tokens[this.index + offset]; }
    
    public convert(sourceCode: string): string {
        this.tokens = tokenize(sourceCode);
        this.index = -1;

        // console.log(this.tokens);
        let code = "";

        
        while (this.eat().type !== TokenType.EOF) {
            let token = this.current();
            const type = token.type;
            const value = token.value;
    
            switch(type) {
                case TokenType.Uint:
                case TokenType.Int:
                case TokenType.Float:
                    const wgslType = TypeToWGSL[value];
                    const previous = this.previous();
                    const next = this.next();

                    // Function
                    if (previous && (previous.type === TokenType.OpenParen || previous.type === TokenType.Comma)) {
                        const previousPrevious = this.previous(2);
                        const varName = this.eat().value;
                        // Pointers
                        if (next.type === TokenType.Asterisk) {
                            const realVarName = this.eat();
                            code += `${realVarName.value}: ${wgslType}`;
                        }
                        // Loops
                        else if (previousPrevious && previousPrevious.type === TokenType.ForLoop) {
                            code += `var ${varName}`;
                        }
                        else {
                            code += `${varName}: ${wgslType}`;
                        }
                    }
                    // Casting
                    else if (next && next.type === TokenType.OpenParen) {
                        code += `${wgslType}`;
                    }
                    // Variable declaration
                    else {
                        const varName = this.next().value;
                        code += `var ${varName}: ${wgslType}`;
                    }
                    break;

                // case TokenType.Semicolon:
                case TokenType.Number:
                case TokenType.OpenParen:
                case TokenType.CloseParen:
                case TokenType.Comma:
                case TokenType.Dot:
                case TokenType.OpenBracket:
                case TokenType.CloseBracket:
                // case TokenType.OpenBrace:
                case TokenType.CloseBrace:
                case TokenType.NewLine:
                case TokenType.Comment:
                    code += `${value}`;
                    break

                case TokenType.GreaterThan:
                case TokenType.LessThan:
                    code += ` ${value} `;
                    break

                case TokenType.Equals:
                case TokenType.Asterisk:
                case TokenType.BinaryOperator: {
                    const next = this.next();
                    const previous = this.previous();
                    // Handles ++ -- etc
                    if ((next && next.type === TokenType.BinaryOperator) || (previous && previous.type === TokenType.BinaryOperator)) {
                        code += `${value}`;
                    }
                    else {
                        code += ` ${value} `;
                    }
                    break
                }

                case TokenType.OpenBrace:
                    code += ` ${value}`;
                    break

                case TokenType.ForLoop:
                case TokenType.WhileLoop:
                case TokenType.IfCondition:
                {
                    if (this.next().type !== TokenType.OpenParen) throw Error("Loop need to be followed by an open parenthesis.");
                    code += ` ${value}`;
                    break
                }

                case TokenType.Semicolon:
                    if (this.next().type === TokenType.NewLine) code += `${value}`;
                    else code += `${value} `;
                    break

                case TokenType.Identifier: {
                    // Handles var creation uint id = 10; == var id: u32 = 10; need to remove id
                    const previous = this.previous();
                    if (previous && (previous.type === TokenType.Uint || previous.type === TokenType.Int || previous.type === TokenType.Float)) {
                        break;
                    }
                    // TODO: Need to specifiy the return type
                    // Probably easiest is to inject another token at the end of the function?
                    if (value === "void") code += `fn `;
                    else {
                        code += `${value}`;
                    }

                    break
                }

                case TokenType.Const: {
                    code += `${value} `;
                    break;
                }

                case TokenType.Shared:
                case TokenType.Extern: {
                    console.error(token);
                    throw Error("Not implemented.");
                }


                default:
                    console.error("Code converter doesn't know about token", token);
                    throw Error("Error converting code");
            }
        }

        console.log("code", code);

        return code;
    }
}