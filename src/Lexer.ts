// https://github.com/tlaceby/guide-to-interpreters-series
// -----------------------------------------------------------
// ---------------          LEXER          -------------------
// ---  Responsible for producing tokens from the source   ---
// -----------------------------------------------------------

// Represents tokens that our language understands in parsing.
export enum TokenType {
    // Literal Types
    Number,
    Identifier,
    
    // Keywords
    Int,
    Uint,
    Float,
    FloatPtr,

    // Grouping * Operators
    UnaryOperator,
    BinaryOperator,

    RelationalOperator,
    EqualityOperator,

    LogicalOperator,
    CompoundOperator,
    
    Equals,
    Ampersand,
    Pipe,

    Comma,
    Dot,
    Colon,
    Semicolon,
    OpenParen, // (
    CloseParen, // )
    OpenBrace, // {
    CloseBrace, // }
    OpenBracket, // [
    CloseBracket, //]

    // Relational operators
    LessThan,
    LessThanRqual,
    GreaterThan,
    GreaterThanEqual,
    EqualTo,

    // Loops
    ForLoop,
    WhileLoop,

    // Conditions
    IfCondition,

    // Pointers
    Asterisk,

    // Specials
    Void,
    Define,
    Extern,
    Shared,
    Const,

    // Function qualifiers
    Global,
    
    // Comments, why not
    Comment,

    NewLine,
    EOF, // Signified the end of file
}

/**
 * Constant lookup for keywords and known identifiers + symbols.
 */
const KEYWORDS: Record<string, TokenType> = {
    int: TokenType.Int,
    uint: TokenType.Uint,
    float: TokenType.Float,

    void: TokenType.Void,
    for: TokenType.ForLoop,
    while: TokenType.WhileLoop,
    if: TokenType.IfCondition,

    "#define": TokenType.Define,
    __global__: TokenType.Global,
    __shared__: TokenType.Shared,
    extern: TokenType.Extern,
    const: TokenType.Const,
};

// Reoresents a single token from the source-code.
export interface Token {
    value: string; // contains the raw value as seen inside the source code.
    type: TokenType; // tagged structure.
}

// Returns a token of a given type and value
function token(value = "", type: TokenType): Token {
    return { value, type };
}

/**
 * Returns whether the string passed is [a-zA-Z0-9_] (a valid variable name)
 */
function isword(src: string) {
    // return src.toUpperCase() != src.toLowerCase();
    return /^[a-zA-Z0-9_#]+$/.test(src);
}

/**
 * Returns true if the character is whitespace like -> [\s, \t, \n]
 */
function isskippable(str: string) {
    return str == " " || str == "\n" || str == "\t" || str == "\r";
    // return str == " " || str == "\t" || str == "\r";
}

/**
 Return whether the character is a valid integer -> [0-9]
 */
function isint(str: string) {
    const c = str.charCodeAt(0);
    const bounds = ["0".charCodeAt(0), "9".charCodeAt(0)];
    return c >= bounds[0] && c <= bounds[1];
}

/**
 * Given a string representing source code: Produce tokens and handles
 * possible unidentified characters.
 *
 * - Returns a array of tokens.
 * - Does not modify the incoming string.
 */
export function tokenize(sourceCode: string): Token[] {
    const tokens = new Array<Token>();
    const src = sourceCode.split("");

    // produce tokens until the EOF is reached.
    while (src.length > 0) {
        // Handle comments
        if (src[0] == "/") {
            const isComment = src[1] === "/" || src[1] === "*";
            const commentType = isComment && src[1] === "/" ? "single" : "multi";
            if (isComment) {
                let comment = "";
                while (src.length > 0) {
                    comment += src.shift();
                    if (commentType === "single" && comment[comment.length - 1] === "\n") break;
                    else if (commentType === "multi" && comment[comment.length - 2] === "*" && comment[comment.length - 1] === "/") break;
                }
                if (commentType == "multi") comment.slice(0, comment.length - 4);
                tokens.push(token(comment, TokenType.Comment));
            }
            else {
                tokens.push(token(src.shift(), TokenType.Asterisk));
            }
        }

        // BEGIN PARSING ONE CHARACTER TOKENS
        else if (src[0] == "(") tokens.push(token(src.shift(), TokenType.OpenParen));
        else if (src[0] == ")") tokens.push(token(src.shift(), TokenType.CloseParen));
        else if (src[0] == "{") tokens.push(token(src.shift(), TokenType.OpenBrace));
        else if (src[0] == "}") tokens.push(token(src.shift(), TokenType.CloseBrace));
        else if (src[0] == "[") tokens.push(token(src.shift(), TokenType.OpenBracket));
        else if (src[0] == "]") tokens.push(token(src.shift(), TokenType.CloseBracket));

        // Unary
        else if (src[0] == "+" && src[1] && src[1] === "+") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.UnaryOperator));
        else if (src[0] == "-" && src[1] && src[1] === "-") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.UnaryOperator));
        // Relational
        else if (src[0] == ">" && src[1] && src[1] === "=") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.RelationalOperator));
        else if (src[0] == "<" && src[1] && src[1] === "=") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.RelationalOperator));
        // Equality
        else if (src[0] == "=" && src[1] && src[1] === "=") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.EqualityOperator));
        // Logical
        else if (src[0] == "&" && src[1] && src[1] === "&") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.LogicalOperator));
        else if (src[0] == "|" && src[1] && src[1] === "|") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.LogicalOperator));
        // Compound
        else if (src[0] == "+" && src[1] && src[1] === "=") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.CompoundOperator));
        else if (src[0] == "-" && src[1] && src[1] === "=") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.CompoundOperator));
        else if (src[0] == "*" && src[1] && src[1] === "=") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.CompoundOperator));
        else if (src[0] == "/" && src[1] && src[1] === "=") tokens.push(token(`${src.shift()}${src.shift()}`, TokenType.CompoundOperator));

        // HANDLE BINARY OPERATORS
        else if (src[0] == "+" || src[0] == "-" || src[0] == "/" || src[0] == "%") {
            tokens.push(token(src.shift(), TokenType.BinaryOperator));
        }
        // Separate * so it can handle pointers
        else if (src[0] == "*") tokens.push(token(src.shift(), TokenType.Asterisk));

        else if (src[0] == "=") tokens.push(token(src.shift(), TokenType.Equals));
        else if (src[0] == ">") tokens.push(token(src.shift(), TokenType.RelationalOperator));
        else if (src[0] == "<") tokens.push(token(src.shift(), TokenType.RelationalOperator));

        else if (src[0] == ";") tokens.push(token(src.shift(), TokenType.Semicolon));
        else if (src[0] == ":") tokens.push(token(src.shift(), TokenType.Colon));
        else if (src[0] == ",") tokens.push(token(src.shift(), TokenType.Comma));
        else if (src[0] == ".") tokens.push(token(src.shift(), TokenType.Dot));
        // Ampersand & pipe
        else if (src[0] == "&") tokens.push(token(src.shift(), TokenType.Ampersand));
        else if (src[0] == "|") tokens.push(token(src.shift(), TokenType.Pipe));

        // HANDLE MULTICHARACTER KEYWORDS, TOKENS, IDENTIFIERS ETC...
        else {
            // Handle numeric literals -> Integers
            if (isint(src[0])) {
                let num = "";
                while (src.length > 0 && isint(src[0])) {
                    num += src.shift();
                }

                // append new numeric token.
                tokens.push(token(num, TokenType.Number));
            } // Handle Identifier & Keyword Tokens.
            else if (isword(src[0])) {
                let ident = "";
                while (src.length > 0 && isword(src[0])) {
                    ident += src.shift();
                }

                // CHECK FOR RESERVED KEYWORDS
                const reserved = KEYWORDS[ident];

                // If value is not undefined then the identifier is
                // reconized keyword
                if (typeof reserved == "number") tokens.push(token(ident, reserved));
                else tokens.push(token(ident, TokenType.Identifier));
            } else if (isskippable(src[0])) {
                // Skip uneeded chars.
                src.shift();
            } // Handle unreconized characters.
            // TODO: Impliment better errors and error recovery.
            else {
                throw Error(`Unreconized character found in source: ${src[0].charCodeAt(0)} ${src[0]}`);
            }
        }
    }

    tokens.push({ type: TokenType.EOF, value: "EndOfFile" });
    return tokens;
}