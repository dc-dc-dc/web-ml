import { Tensor } from "./tensor";

type TypedArray = Array;
type Dtype = "float32" | "int32" | "uint32";
type Shape = number[];
type Strides = number[];

type OpId =
    "abs" |
    "add" |
    "min" |
    "max" |
    "sub" |
    "mul" |
    "div" |
    "pow" |
    "neg" |
    "round" |
    "ceil" |
    "floor" |
    "log" |
    "exp" |
    "log2" |
    "exp2" |
    "sqrt" |
    "sin" |
    "cos" |
    "tan" |
    "asin" |
    "acos" |
    "atan" |
    "sinh" |
    "cosh" |
    "tanh" |
    "greater" |
    "equal" |
    "reshape" |
    "broadcast";


interface Op {
    eval(inputs: Tensor[]): Promise<TypedArray>;
}