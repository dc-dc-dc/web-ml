import { createVitest } from "vitest/node";
// Watch doesn't work for some reason
const vitest = await createVitest("test", { watch: false });
await vitest.start();
