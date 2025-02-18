import { ReactNode, useRef, useState } from "react";

const SEPARATOR_COLOR = "#f1f1f1"; 
const MIN_WIDTH_PERCENT = 20;
const MAX_WIDTH_PERCENT = 80;

export function ResizablePanels(props: {
    firstComponent: ReactNode,
    secondComponent: ReactNode,
}) {
    const [leftWidth, setLeftWidth] = useState(50);  // percentage 0-100
    const containerRef = useRef<HTMLDivElement | null>(null);

    function onMouseMove(event: MouseEvent) {
        if (containerRef.current === null) return;

        const rect = containerRef.current.getBoundingClientRect();
        const containerWidth = rect.right - rect.left;

        let relativeMousePosition = 100 * (event.clientX - rect.left) / containerWidth;

        if (relativeMousePosition > MAX_WIDTH_PERCENT) {
            relativeMousePosition = MAX_WIDTH_PERCENT;
        } else if (relativeMousePosition < MIN_WIDTH_PERCENT) {
            relativeMousePosition = MIN_WIDTH_PERCENT;
        }
        setLeftWidth(relativeMousePosition);
    }

    function onDividerMouseDown() {
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onDivideMouseUp);
    }

    function onDivideMouseUp() {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onDivideMouseUp);
    }

    return (
        <div ref={containerRef} style={{
            display: "flex",
            // assuming outer container is flexbox, this will occupy all space avaliable
            flex: 1,
        }}>
            <div
                className="content-box"
                style={{ width: `${leftWidth}%`, display: "flex" }}
            >
                {props.firstComponent}
            </div>

            <div
                style={{
                    zIndex: 1,
                    backgroundColor: SEPARATOR_COLOR,
                    width: "8px",
                    cursor: "ew-resize",
                }}
                onMouseDown={onDividerMouseDown}
            />

            <div
                className="content-box"
                style={{ width: `${100 - leftWidth}%`, display: "flex" }}
            >
                {props.secondComponent}
            </div>
        </div>
    )
}
