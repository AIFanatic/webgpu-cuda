var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};

// src-ast/Lexer.ts
var KEYWORDS = {
  int: 2 /* Int */,
  uint: 3 /* Uint */,
  float: 4 /* Float */,
  for: 19 /* ForLoop */,
  while: 20 /* WhileLoop */,
  if: 21 /* IfCondition */,
  __shared__: 24 /* Shared */,
  extern: 23 /* Extern */,
  const: 25 /* Const */
};
function token(value = "", type) {
  return { value, type };
}
function isvariablename(src) {
  return /^[a-zA-Z0-9_]+$/.test(src);
}
function isskippable(str) {
  return str == " " || str == "	" || str == "\r";
}
function isint(str) {
  const c = str.charCodeAt(0);
  const bounds = ["0".charCodeAt(0), "9".charCodeAt(0)];
  return c >= bounds[0] && c <= bounds[1];
}
function tokenize(sourceCode) {
  const tokens = new Array();
  const src = sourceCode.split("");
  while (src.length > 0) {
    if (src[0] == "/") {
      const isComment = src[1] === "/" || src[1] === "*";
      const commentType = isComment && src[1] === "/" ? "single" : "multi";
      if (isComment) {
        let comment = "";
        while (src.length > 0) {
          comment += src.shift();
          if (commentType === "single" && comment[comment.length - 1] === "\n")
            break;
          else if (commentType === "multi" && comment[comment.length - 2] === "*" && comment[comment.length - 1] === "/")
            break;
        }
        if (commentType == "multi")
          comment.slice(0, comment.length - 4);
        tokens.push(token(comment, 26 /* Comment */));
      } else {
        tokens.push(token(src.shift(), 22 /* Asterisk */));
      }
    } else if (src[0] == "(")
      tokens.push(token(src.shift(), 11 /* OpenParen */));
    else if (src[0] == ")")
      tokens.push(token(src.shift(), 12 /* CloseParen */));
    else if (src[0] == "{")
      tokens.push(token(src.shift(), 13 /* OpenBrace */));
    else if (src[0] == "}")
      tokens.push(token(src.shift(), 14 /* CloseBrace */));
    else if (src[0] == "[")
      tokens.push(token(src.shift(), 15 /* OpenBracket */));
    else if (src[0] == "]")
      tokens.push(token(src.shift(), 16 /* CloseBracket */));
    else if (src[0] == "+" || src[0] == "-" || src[0] == "/" || src[0] == "%") {
      tokens.push(token(src.shift(), 5 /* BinaryOperator */));
    } else if (src[0] == "*")
      tokens.push(token(src.shift(), 22 /* Asterisk */));
    else if (src[0] == "=")
      tokens.push(token(src.shift(), 6 /* Equals */));
    else if (src[0] == ";")
      tokens.push(token(src.shift(), 10 /* Semicolon */));
    else if (src[0] == ":")
      tokens.push(token(src.shift(), 9 /* Colon */));
    else if (src[0] == ",")
      tokens.push(token(src.shift(), 7 /* Comma */));
    else if (src[0] == ".")
      tokens.push(token(src.shift(), 8 /* Dot */));
    else if (src[0] == ">")
      tokens.push(token(src.shift(), 18 /* GreaterThan */));
    else if (src[0] == "<")
      tokens.push(token(src.shift(), 17 /* LessThan */));
    else if (src[0] == "\n")
      tokens.push(token(src.shift(), 27 /* NewLine */));
    else {
      if (isint(src[0])) {
        let num = "";
        while (src.length > 0 && isint(src[0])) {
          num += src.shift();
        }
        tokens.push(token(num, 0 /* Number */));
      } else if (isvariablename(src[0])) {
        let ident = "";
        while (src.length > 0 && isvariablename(src[0])) {
          ident += src.shift();
        }
        const reserved = KEYWORDS[ident];
        if (typeof reserved == "number")
          tokens.push(token(ident, reserved));
        else
          tokens.push(token(ident, 1 /* Identifier */));
      } else if (isskippable(src[0])) {
        src.shift();
      } else {
        throw Error(`Unreconized character found in source: ${src[0].charCodeAt(0)} ${src[0]}`);
      }
    }
  }
  tokens.push({ type: 28 /* EOF */, value: "EndOfFile" });
  return tokens;
}

// src-ast/WGSLCodeGenerator.ts
var TypeToWGSL = {
  uint: "u32",
  int: "i32",
  float: "f32"
};
var WGSLCodeGenerator = class {
  constructor() {
    __publicField(this, "tokens");
    __publicField(this, "index");
  }
  eat() {
    return this.tokens[++this.index];
  }
  current() {
    return this.tokens[this.index];
  }
  previous(offset = 1) {
    return this.tokens[this.index - offset];
  }
  next(offset = 1) {
    return this.tokens[this.index + offset];
  }
  convert(sourceCode) {
    this.tokens = tokenize(sourceCode);
    this.index = -1;
    let code = "";
    while (this.eat().type !== 28 /* EOF */) {
      let token2 = this.current();
      const type = token2.type;
      const value = token2.value;
      switch (type) {
        case 3 /* Uint */:
        case 2 /* Int */:
        case 4 /* Float */:
          const wgslType = TypeToWGSL[value];
          const previous = this.previous();
          const next = this.next();
          if (previous && (previous.type === 11 /* OpenParen */ || previous.type === 7 /* Comma */)) {
            const previousPrevious = this.previous(2);
            const varName = this.eat().value;
            if (next.type === 22 /* Asterisk */) {
              const realVarName = this.eat();
              code += `${realVarName.value}: ${wgslType}`;
            } else if (previousPrevious && previousPrevious.type === 19 /* ForLoop */) {
              code += `var ${varName}`;
            } else {
              code += `${varName}: ${wgslType}`;
            }
          } else if (next && next.type === 11 /* OpenParen */) {
            code += `${wgslType}`;
          } else {
            const varName = this.next().value;
            code += `var ${varName}: ${wgslType}`;
          }
          break;
        case 0 /* Number */:
        case 11 /* OpenParen */:
        case 12 /* CloseParen */:
        case 7 /* Comma */:
        case 8 /* Dot */:
        case 15 /* OpenBracket */:
        case 16 /* CloseBracket */:
        case 14 /* CloseBrace */:
        case 27 /* NewLine */:
        case 26 /* Comment */:
          code += `${value}`;
          break;
        case 18 /* GreaterThan */:
        case 17 /* LessThan */:
          code += ` ${value} `;
          break;
        case 6 /* Equals */:
        case 22 /* Asterisk */:
        case 5 /* BinaryOperator */: {
          const next2 = this.next();
          const previous2 = this.previous();
          if (next2 && next2.type === 5 /* BinaryOperator */ || previous2 && previous2.type === 5 /* BinaryOperator */) {
            code += `${value}`;
          } else {
            code += ` ${value} `;
          }
          break;
        }
        case 13 /* OpenBrace */:
          code += ` ${value}`;
          break;
        case 19 /* ForLoop */:
        case 20 /* WhileLoop */:
        case 21 /* IfCondition */: {
          if (this.next().type !== 11 /* OpenParen */)
            throw Error("Loop need to be followed by an open parenthesis.");
          code += ` ${value}`;
          break;
        }
        case 10 /* Semicolon */:
          if (this.next().type === 27 /* NewLine */)
            code += `${value}`;
          else
            code += `${value} `;
          break;
        case 1 /* Identifier */: {
          const previous2 = this.previous();
          if (previous2 && (previous2.type === 3 /* Uint */ || previous2.type === 2 /* Int */ || previous2.type === 4 /* Float */)) {
            break;
          }
          if (value === "void")
            code += `fn `;
          else {
            code += `${value}`;
          }
          break;
        }
        case 25 /* Const */: {
          code += `${value} `;
          break;
        }
        case 24 /* Shared */:
        case 23 /* Extern */: {
          console.error(token2);
          throw Error("Not implemented.");
        }
        default:
          console.error("Code converter doesn't know about token", token2);
          throw Error("Error converting code");
      }
    }
    console.log("code", code);
    return code;
  }
};
export {
  WGSLCodeGenerator
};
