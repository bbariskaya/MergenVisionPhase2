import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar.tsx'
import styles from './appLayout.module.css'

export default function AppLayout() {
  return (
    <div className={styles.layout}>
      <Sidebar />
      <main className={styles.main}>
        <div className={styles.content}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
