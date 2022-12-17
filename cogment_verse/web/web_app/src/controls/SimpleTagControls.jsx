import { useCallback, useState } from "react";
import { cogment_verse } from "../data_pb";
import { useDocumentKeypressListener, usePressedKeys } from "../hooks/usePressedKeys";
import { useRealTimeUpdate } from "../hooks/useRealTimeUpdate";
import { TEACHER_ACTOR_CLASS, TEACHER_NOOP_ACTION } from "../utils/constants";
import { DPad, usePressedButtons, DPAD_BUTTONS } from "../components/DPad";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";

const DEACTIVATED_BUTTONS_TEACHER = [];
const DEACTIVATED_BUTTONS_PLAYER = [DPAD_BUTTONS.UP];

export const SimpleTagEnvironments = ["environments.mpe_adapter.Environment/pettingzoo.mpe.simple_tag_v2"];

export const SimpleTagControls = ({ sendAction, fps = 20, actorClass, ...props }) => {
  const isTeacher = actorClass === TEACHER_ACTOR_CLASS;
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();
  const { pressedButtons, isButtonPressed, setPressedButtons } = usePressedButtons();
  const [activeButtons, setActiveButtons] = useState([]);

  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.has("ArrowRight") || isButtonPressed(DPAD_BUTTONS.RIGHT)) {
        setActiveButtons([DPAD_BUTTONS.RIGHT]);
        sendAction(new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 1 }] } }));
        return;
      } else if (pressedKeys.has("ArrowDown") || isButtonPressed(DPAD_BUTTONS.DOWN)) {
        setActiveButtons([DPAD_BUTTONS.DOWN]);
        sendAction(new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 2 }] } }));
        return;
      } else if (pressedKeys.has("ArrowLeft") || isButtonPressed(DPAD_BUTTONS.LEFT)) {
        setActiveButtons([DPAD_BUTTONS.LEFT]);
        sendAction(new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 3 }] } }));
        return;
      } else if (isTeacher) {
        if (pressedKeys.has("ArrowUp") || isButtonPressed(DPAD_BUTTONS.UP)) {
          setActiveButtons([DPAD_BUTTONS.UP]);
          sendAction(new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 0 }] } }));
          return;
        }
        setActiveButtons([]);
        sendAction(TEACHER_NOOP_ACTION);
        return;
      }
      setActiveButtons([]);
      sendAction(new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 0 }] } }));
    },
    [isButtonPressed, pressedKeys, sendAction, setActiveButtons, isTeacher]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div {...props}>
      <div className="flex flex-row p-5 justify-center">
        <DPad
          pressedButtons={pressedButtons}
          onPressedButtonsChange={setPressedButtons}
          activeButtons={activeButtons}
          disabled={paused || (isTeacher ? DEACTIVATED_BUTTONS_TEACHER : DEACTIVATED_BUTTONS_PLAYER)}
        />
      </div>
      <div className="flex flex-row gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right Arrows", "Fire left/right engine"],
          ["Down Arrow", "Fire the main engine"],
          isTeacher ? ["Up Arrow", "turn off engine"] : null,
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};