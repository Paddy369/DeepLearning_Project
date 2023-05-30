import MileStoneSelection from "../MilestoneSelection"
import ModelSelection from "../ModelSelection"

const Navbar = ({ setMilestone, setModel }) => {
    return (
        <div>
            <MileStoneSelection setMilestone={setMilestone} />
            <ModelSelection setModel={setModel} />
        </div>
    )
}

export default Navbar