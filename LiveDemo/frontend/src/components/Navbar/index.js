import MileStoneSelection from "../MilestoneSelection"
import ModelSelection from "../ModelSelection"

const Navbar = ({ milestone, setMilestone, model, setModel }) => {
    return (
        <div>
            <MileStoneSelection milestone={milestone} setMilestone={setMilestone} />
            <ModelSelection milestone={milestone} model={model} setModel={setModel} />
        </div>
    )
}

export default Navbar