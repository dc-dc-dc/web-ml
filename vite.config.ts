import {defineConfig} from "vite";

export default defineConfig(({command, mode }) => {
    const isDev = mode == "development" || mode == "test";
    return {
        define: {
            __DEV__: isDev
        }
    }
});