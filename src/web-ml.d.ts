import { Tensor } from "./tensor";

type TypedArray = Array;

type Dtype = "float32" | "int32" | "uint32";
type Shape = number[];
type Tensor = {};
interface Op {
    eval(inputs: TypedArray[]): Promise<TypedArray>;
}