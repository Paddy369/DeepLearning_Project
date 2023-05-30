import styles from './styles.module.css'
import { useState } from 'react'

const ImageInputBox = ({ setImage }) => {

    const [text, setText] = useState("Drop the Image here")
    const [path, setPath] = useState("")

    const handleDrop = e => {
        e.preventDefault()
        const image = e.dataTransfer.files[0]
        setText("")
        setPath("./img/" + image.name)
        setImage(image.name)
    }

    const handleDragOver = e => {
        e.preventDefault()
    }

    return (
        <div onDragOver={handleDragOver} onDrop={handleDrop} className={styles.box} style={{backgroundImage: `url(${path})` }}>
            {text}
        </div>
    )
}

export default ImageInputBox