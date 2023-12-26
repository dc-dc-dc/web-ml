import type { Dtype } from "./web-ml";

export function getDtypeSize(dtype: Dtype): number {
    switch (dtype) {
        case "float32":
        case "int32":
        case "uint32":
            return 4;
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
}

export const int32: Dtype = "int32";
export const uint32: Dtype = "uint32";
export const float32: Dtype = "float32";