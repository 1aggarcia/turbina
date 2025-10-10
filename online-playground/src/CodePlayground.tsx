import { Editor } from "@monaco-editor/react";
import { CSSProperties, useState } from "react";
import { run_turbina_program } from "turbina";
import { Header } from "./Header";
import { ResizablePanels } from "./ResizablePanels";
import { SAMPLE_PROGRAMS } from "./samplePrograms";

const CODE_FONT_SIZE = 13;

const containerStyle: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    fontSize: CODE_FONT_SIZE,
    height: "100%",
    width: "100%",
};

const consoleDivStyle: CSSProperties = {
    fontSize: CODE_FONT_SIZE,
    textAlign: "left",
    padding: "1rem",
};

const getRandomSampleProgram = () => {
    const randIdx = Math.floor(Math.random() * SAMPLE_PROGRAMS.length)
    return SAMPLE_PROGRAMS[randIdx];
}

/** Should not be mounted until the web assembly has been initialized */
export function CodePlayground() {
    const [sourceCode, setSourceCode] = useState(getRandomSampleProgram());
    const [consoleOutput, setConsoleOutput] = useState("");

    function writeToConsole(data: string) {
        // state change must be queued with a callback since the WASM runtime
        // doesn't wait for React updates
        setConsoleOutput(output => output + data);
    }

    function onExecuteClick() {
        try {
            run_turbina_program(
                sourceCode, writeToConsole, writeToConsole);
            writeToConsole("\n");
        } catch (error) {
            writeToConsole(`${error}\n`);
            console.error(error);
        }
    }

    function onClearClick() {
        setConsoleOutput("");
    }

    return (
        <div style={containerStyle}>
            <Header>
                <button className="round-left" onClick={onExecuteClick}>
                    Execute
                </button>
                <button className="round-right" onClick={onClearClick}>
                    Clear Output
                </button>
            </Header>

            <ResizablePanels
                firstComponent={
                    <div style={{ paddingBlock: "10px", width: "100%" }}>
                        <Editor
                            value={sourceCode}
                            /* Rust syntax highlighting works decently and
                            doesn't give validation errors */
                            language="rust"
                            onChange={text => setSourceCode(text ?? "")}
                            options={{
                                minimap: { enabled: false },
                                fontSize: CODE_FONT_SIZE,
                                contextmenu: false,
                                scrollBeyondLastLine: false,
                            }}
                        />
                    </div>
                }
                secondComponent={
                    <pre style={{ ...consoleDivStyle, margin: 0 }}>
                        {consoleOutput}
                    </pre>
                }
            />
        </div>
    )
}
